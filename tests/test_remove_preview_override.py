"""Remove Preview Override contract: samplers using its output don't run the named hook.

Hooks are seen the way a sampler sees them: ComfyUI's sampler prepares the
model and runs the wrappers and callbacks it gathers.
"""
import torch

import comfy.model_patcher
import comfy.patcher_extension
import comfy.sampler_helpers
from comfy.patcher_extension import CallbacksMP, WrappersMP

import scripts_nodes

PREVIEW = "kj_preview_override"
HOOK_TYPES = (WrappersMP.OUTER_SAMPLE, WrappersMP.DIFFUSION_MODEL)


def _model_with_hooks(ran, *names):
    cpu = torch.device("cpu")
    model = comfy.model_patcher.ModelPatcher(torch.nn.Linear(1, 1), load_device=cpu, offload_device=cpu)
    for name in names:
        for hook_type in HOOK_TYPES:
            def wrapper(executor, *args, name=name, hook_type=hook_type, **kwargs):
                ran.append((name, hook_type))
                return executor(*args, **kwargs)
            model.add_wrapper_with_key(hook_type, name, wrapper)
        model.add_callback_with_key(CallbacksMP.ON_PRE_RUN, name, lambda *args, name=name: ran.append((name, "callback")))
    return model


def _sample(model):
    """Run every wrapper and callback a sampler would gather from ``model``."""
    options = {"transformer_options": {}}
    comfy.sampler_helpers.prepare_model_patcher(model, {}, options)
    for hook_type in HOOK_TYPES:
        wrappers = comfy.patcher_extension.get_all_wrappers(hook_type, options, is_model_options=True)
        comfy.patcher_extension.WrapperExecutor.new_executor(lambda: None, wrappers).execute()
    for callback in comfy.patcher_extension.get_all_callbacks(CallbacksMP.ON_PRE_RUN, options["transformer_options"]):
        callback(model)


def _remove(model, name=PREVIEW):
    return scripts_nodes.NODE_CLASS_MAPPINGS["RemovePreviewOverride"]().remove(model, name)[0]


def test_default_name_is_kjnodes_preview_override():
    inputs = scripts_nodes.NODE_CLASS_MAPPINGS["RemovePreviewOverride"].INPUT_TYPES()["required"]
    assert list(inputs) == ["model", "name"]
    assert inputs["name"][1]["default"] == PREVIEW


def test_output_model_samples_without_the_named_hook():
    ran = []
    model = _model_with_hooks(ran, PREVIEW, "other")

    _sample(_remove(model))
    assert {name for name, _ in ran} == {"other"}
    assert len(ran) == len(HOOK_TYPES) + 1


def test_input_model_keeps_the_hook():
    ran = []
    model = _model_with_hooks(ran, PREVIEW)
    _remove(model)

    _sample(model)
    assert sorted(ran, key=str) == sorted([(PREVIEW, t) for t in HOOK_TYPES] + [(PREVIEW, "callback")], key=str)


def test_another_name_leaves_the_hook_in_place():
    ran = []
    model = _model_with_hooks(ran, PREVIEW)

    _sample(_remove(model, "renamed_preview"))
    assert len(ran) == len(HOOK_TYPES) + 1
