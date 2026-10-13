"""H3 LoRA on Audio or Video Tokens contract.

The LoRA's change reaches only the chosen stream's rows of each LoRA'd layer;
the other rows get exactly the base layer's output. Seen through the model's
block-replace path, the same way a sampler runs the model.
"""
import types

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

import comfy.model_patcher
import folder_paths
from comfy.ldm.minimax.model import MiniMaxH3Model

import scripts_nodes

LAYERS = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")


class Base(nn.Module):
    """Stands in for ComfyUI's BaseModel: the transformer under ``diffusion_model``."""

    def __init__(self, diffusion_model):
        super().__init__()
        self.diffusion_model = diffusion_model
        self.model_config = types.SimpleNamespace(unet_config={})


def _h3(num_layers=2):
    torch.manual_seed(0)
    model = MiniMaxH3Model(
        hidden_size=96, num_layers=num_layers, token_refiner_num_layers=0,
        num_attention_heads=1, attention_head_dim=96, ffn_hidden_size=32,
        latents_dim=2, audio_latents_dim=3, text_dim=5,
        timestep_input_dim=8, time_embed_hidden_size=8, time_embed_dim=8,
        dtype=torch.float32, device="cpu", operations=nn,
    )
    for p in model.parameters():
        nn.init.normal_(p, std=0.2)
    return model.requires_grad_(False)


def _patcher(diffusion_model):
    cpu = torch.device("cpu")
    return comfy.model_patcher.ModelPatcher(Base(diffusion_model), load_device=cpu, offload_device=cpu)


@pytest.fixture
def lora(tmp_path):
    """A rank-4 LoRA on every block layer of the tiny model, plus a layer outside the blocks."""
    torch.manual_seed(1)
    model = _h3()
    sd = {}
    for i in range(len(model.blocks)):
        for path in LAYERS:
            layer = model.blocks[i].get_submodule(path)
            key = f"diffusion_model.blocks.{i}.{path}"
            sd[f"{key}.lora_A.weight"] = torch.randn(4, layer.in_features) * 0.3
            sd[f"{key}.lora_B.weight"] = torch.randn(layer.out_features, 4) * 0.3
            sd[f"{key}.alpha"] = torch.tensor(2.0)
    sd["diffusion_model.final_layer.audio_out.lora_A.weight"] = torch.randn(4, 96)
    sd["diffusion_model.final_layer.audio_out.lora_B.weight"] = torch.randn(3 * 2, 4)
    folder = tmp_path / "loras"
    folder.mkdir()
    save_file(sd, str(folder / "test_lora.safetensors"))
    folder_paths.add_model_folder_path("loras", str(folder))
    return "test_lora.safetensors", sd


def _apply(patcher, name, strength=1.0, tokens="audio"):
    return scripts_nodes.NODE_CLASS_MAPPINGS["H3TokenLora"]().apply(patcher, name, strength, tokens)[0]


def _run(model, patcher):
    torch.manual_seed(2)
    x = [torch.randn(1, 2, 1, 4, 6), torch.randn(1, 3, 2, 2)]
    with torch.inference_mode():
        return model(x, torch.tensor([500.0]), torch.randn(1, 4, 96),
                     transformer_options=patcher.model_options.get("transformer_options", {}),
                     minimax_payload={"audio_scale": 1.0})


@pytest.mark.parametrize("tokens", ["audio", "video"])
def test_lora_reaches_only_the_chosen_stream_rows(lora, tokens):
    name, sd = lora
    model = _h3()
    seen = {}
    base_patcher = _patcher(model)

    def hook(module, inputs, output):
        # Clone now: attention applies RoPE in place to this output afterwards.
        seen["x"], seen["out"] = inputs[0].clone(), output.clone()

    def record(args, extra):
        # Registered inside the LoRA node's patch, so it runs after the node's hook.
        seen["segments"] = args["layout"].segments
        handle = model.blocks[0].attn.qkv_proj.register_forward_hook(hook)
        try:
            return extra["original_block"](args)
        finally:
            handle.remove()

    # Another node's patch on the same block; the LoRA node must keep calling it.
    base_patcher.set_model_patch_replace(record, "dit", "double_block", 0)
    _run(model, _apply(base_patcher, name, tokens=tokens))

    layer = model.blocks[0].attn.qkv_proj
    base = nn.functional.linear(seen["x"], layer.weight, layer.bias)
    key = "diffusion_model.blocks.0.attn.qkv_proj"
    delta = seen["x"] @ sd[f"{key}.lora_A.weight"].t() @ sd[f"{key}.lora_B.weight"].t() * (2.0 / 4)
    chosen = {"audio": {"audio"}, "video": {"video"}}[tokens]
    for a, b, kind in seen["segments"]:
        expected = base[a:b] + delta[a:b] if kind in chosen else base[a:b]
        torch.testing.assert_close(seen["out"][a:b], expected, msg=f"{kind} rows")
    assert any(kind in chosen for _, _, kind in seen["segments"])


def test_hooks_are_gone_after_each_step(lora):
    model = _h3()
    _run(model, _apply(_patcher(model), lora[0]))
    for block in model.blocks:
        for path in LAYERS:
            assert not block.get_submodule(path)._forward_hooks


def test_strength_zero_leaves_the_output_unchanged(lora):
    model = _h3()
    base = _run(model, _patcher(model))
    patched = _run(model, _apply(_patcher(model), lora[0], strength=0.0))
    for a, b in zip(base, patched):
        torch.testing.assert_close(a, b)


def test_audio_tokens_change_the_audio_output(lora):
    model = _h3()
    base = _run(model, _patcher(model))
    patched = _run(model, _apply(_patcher(model), lora[0], tokens="audio"))
    assert not torch.allclose(base[1], patched[1])


def test_refuses_a_model_that_is_not_h3(lora):
    cpu = torch.device("cpu")
    other = comfy.model_patcher.ModelPatcher(Base(nn.Linear(2, 2)), load_device=cpu, offload_device=cpu)
    with pytest.raises(ValueError, match="not MiniMax H3"):
        _apply(other, lora[0])


def test_refuses_a_lora_without_block_layers(tmp_path):
    folder = tmp_path / "loras_empty"
    folder.mkdir()
    save_file({"diffusion_model.final_layer.audio_out.lora_A.weight": torch.randn(4, 96),
               "diffusion_model.final_layer.audio_out.lora_B.weight": torch.randn(6, 4)},
              str(folder / "outside_only.safetensors"))
    folder_paths.add_model_folder_path("loras", str(folder))
    model = _h3()
    with pytest.raises(ValueError, match="no LoRA layers in the H3 transformer blocks"):
        _apply(_patcher(model), "outside_only.safetensors")
