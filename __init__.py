from comfy.comfy_types import ComfyNodeABC

from . import enumerator, h3_token_lora, latent_cache, remove_preview_override

NODE_CLASS_MAPPINGS: dict[str, type[ComfyNodeABC]] = {
    **enumerator.NODE_CLASS_MAPPINGS,
    **h3_token_lora.NODE_CLASS_MAPPINGS,
    **latent_cache.NODE_CLASS_MAPPINGS,
    **remove_preview_override.NODE_CLASS_MAPPINGS,
}
NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    **enumerator.NODE_DISPLAY_NAME_MAPPINGS,
    **h3_token_lora.NODE_DISPLAY_NAME_MAPPINGS,
    **latent_cache.NODE_DISPLAY_NAME_MAPPINGS,
    **remove_preview_override.NODE_DISPLAY_NAME_MAPPINGS,
}
WEB_DIRECTORY: str = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
