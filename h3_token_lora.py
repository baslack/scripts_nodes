import logging
import re

import torch

import comfy.lora
import comfy.utils
import comfy.weight_adapter
import folder_paths
from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict
from comfy.ldm.minimax.model import MiniMaxH3Model

# Row kinds in MiniMax H3's packed token sequence, by stream.
ROW_KINDS = {
    "audio": {"audio", "cond_audio", "ref_audio"},
    "video": {"video", "cond", "ref_img"},
}
BLOCK_LAYER = re.compile(r"^diffusion_model\.blocks\.(\d+)\.(.+)\.weight$")


class TokenRowsPatch:
    """Block-replace patch that applies a LoRA to one stream's rows of an H3 block.

    H3 runs video and audio tokens through the same weights, so a LoRA merged
    into them changes both. This patch leaves the weights alone and, while the
    block runs, adds each LoRA'd layer's low-rank change only to the rows of
    the chosen stream.
    """

    def __init__(self, block, layers, tokens, previous=None):
        """
        Args:
            block: The H3 transformer block.
            layers: ``(path, down, up, scale)`` for each LoRA'd layer, ``path``
                relative to the block.
            tokens: ``"audio"`` or ``"video"``.
            previous: A block-replace patch already set for this block, run in
                place of the original block.
        """
        self.block = block
        self.layers = layers
        self.kinds = ROW_KINDS[tokens]
        self.previous = previous

    def to(self, device_or_dtype):
        """Move the LoRA weights with the model; ComfyUI calls this when it loads and unloads it.

        Dtype casts are skipped: the model dtype can be a storage format such as int8.
        """
        if isinstance(device_or_dtype, torch.device):
            self.layers = [(path, down.to(device_or_dtype), up.to(device_or_dtype), scale)
                           for path, down, up, scale in self.layers]
        if hasattr(self.previous, "to"):
            self.previous = self.previous.to(device_or_dtype)
        return self

    def __call__(self, args, extra):
        spans = [(a, b) for a, b, kind in args["layout"].segments if kind in self.kinds]
        hooks = [
            self.block.get_submodule(path).register_forward_hook(_low_rank_hook(down, up, scale, spans))
            for path, down, up, scale in self.layers
        ]
        try:
            return self.previous(args, extra) if self.previous is not None else extra["original_block"](args)
        finally:
            for hook in hooks:
                hook.remove()


def _low_rank_hook(down, up, scale, spans):
    def hook(module, inputs, output):
        x = inputs[0]
        down_x, up_x = down.to(x.device, x.dtype), up.to(x.device, x.dtype)
        # Tokens are the second-to-last axis of the packed sequence.
        for a, b in spans:
            output[..., a:b, :] += (x[..., a:b, :] @ down_x.t()) @ up_x.t() * scale
        return output

    return hook


class H3TokenLora(ComfyNodeABC):
    """Apply a LoRA to only the audio, or only the video, tokens of a MiniMax H3 model."""

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        """Declare the node inputs.

        Returns:
            The ``model``, the ``lora_name``, its ``strength`` and the
            ``tokens`` it applies to.
        """
        return {
            "required": {
                "model": (IO.MODEL, {}),
                "lora_name": (folder_paths.get_filename_list("loras"), {}),
                "strength": (IO.FLOAT, {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "tokens": (["audio", "video"], {"default": "audio", "tooltip": "The stream whose tokens the LoRA changes."}),
            }
        }

    RETURN_TYPES = (IO.MODEL,)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "loaders"

    def apply(self, model, lora_name: str, strength: float, tokens: str) -> tuple:
        """Patch a clone of ``model`` so the LoRA reaches only one stream's tokens.

        Args:
            model: A MiniMax H3 model patcher.
            lora_name: A LoRA file in ``models/loras``.
            strength: Multiplier on the LoRA's change.
            tokens: ``"audio"`` or ``"video"``.

        Returns:
            The patched clone.

        Raises:
            ValueError: If the model isn't MiniMax H3, or the LoRA has no
                plain LoRA layers in its transformer blocks.
        """
        if not isinstance(model.get_model_object("diffusion_model"), MiniMaxH3Model):
            raise ValueError("H3 Token LoRA: the model is not MiniMax H3.")
        lora = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("loras", lora_name), safe_load=True)
        loaded = comfy.lora.load_lora(lora, comfy.lora.model_lora_keys_unet(model.model, {}))

        per_block = {}
        left_out = 0
        for key, adapter in loaded.items():
            match = BLOCK_LAYER.match(key) if isinstance(key, str) else None
            if match is None:
                left_out += 1
                continue
            if not isinstance(adapter, comfy.weight_adapter.LoRAAdapter) or adapter.weights[3] is not None or adapter.weights[4] is not None:
                raise ValueError(f"H3 Token LoRA: {key} is not a plain LoRA layer (LoCon, DoRA, LoHa and LoKr aren't supported).")
            up, down, alpha = adapter.weights[:3]
            scale = strength * (alpha / down.shape[0] if alpha is not None else 1.0)
            per_block.setdefault(int(match.group(1)), []).append((match.group(2), down, up, scale))
        if not per_block:
            raise ValueError(f"H3 Token LoRA: {lora_name} has no LoRA layers in the H3 transformer blocks.")
        if left_out:
            logging.info(f"H3 Token LoRA: left out {left_out} layers outside the transformer blocks, such as the text refiner.")

        m = model.clone()
        blocks = m.get_model_object("diffusion_model.blocks")
        existing = m.model_options.get("transformer_options", {}).get("patches_replace", {}).get("dit", {})
        for index, layers in per_block.items():
            patch = TokenRowsPatch(blocks[index], layers, tokens, existing.get(("double_block", index)))
            m.set_model_patch_replace(patch, "dit", "double_block", index)
        return (m,)


NODE_CLASS_MAPPINGS: dict[str, type[ComfyNodeABC]] = {
    "H3TokenLora": H3TokenLora,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "H3TokenLora": "H3 LoRA on Audio or Video Tokens",
}
