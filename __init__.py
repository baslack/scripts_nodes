from comfy.comfy_types import ComfyNodeABC

from . import enumerator

NODE_CLASS_MAPPINGS: dict[str, type[ComfyNodeABC]] = {**enumerator.NODE_CLASS_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {**enumerator.NODE_DISPLAY_NAME_MAPPINGS}
WEB_DIRECTORY: str = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
