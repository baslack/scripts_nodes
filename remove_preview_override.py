from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict


class RemovePreviewOverride(ComfyNodeABC):
    """Remove a named hook, such as KJNodes' Model Preview Override, from a model.

    The hook is removed from a clone, so samplers using the original model
    keep it.
    """

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        """Declare the node inputs.

        Returns:
            The ``model`` and the ``name`` of the hook to remove.
        """
        return {
            "required": {
                "model": (IO.MODEL, {}),
                "name": (IO.STRING, {"default": "kj_preview_override"}),
            }
        }

    RETURN_TYPES = (IO.MODEL,)
    RETURN_NAMES = ("model",)
    FUNCTION = "remove"
    CATEGORY = "utils"

    def remove(self, model, name: str) -> tuple:
        """Clone the model without the wrappers and callbacks registered as ``name``.

        Args:
            model: The model patcher.
            name: The key the hook was registered with.

        Returns:
            The clone. Removing a name the model doesn't have changes nothing.
        """
        m = model.clone()
        for wrapper_type in m.wrappers:
            m.remove_wrappers_with_key(wrapper_type, name)
        for call_type in m.callbacks:
            m.remove_callbacks_with_key(call_type, name)
        return (m,)


NODE_CLASS_MAPPINGS: dict[str, type[ComfyNodeABC]] = {
    "RemovePreviewOverride": RemovePreviewOverride,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "RemovePreviewOverride": "Remove Preview Override",
}
