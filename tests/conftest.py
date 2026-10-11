"""Load ComfyUI on the CPU and make this package importable as ``scripts_nodes``."""
import os
import sys
from pathlib import Path

comfyui_path = os.environ.get("COMFYUI_PATH")
if not comfyui_path:
    raise RuntimeError("Set COMFYUI_PATH to a ComfyUI checkout to run these tests.")
sys.path.insert(0, comfyui_path)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Prime ComfyUI's CLI args with --cpu before anything imports comfy.model_management.
original_argv = sys.argv[:]
try:
    sys.argv[:] = [original_argv[0], "--cpu"]
    import comfy.options

    comfy.options.enable_args_parsing()
    import comfy.cli_args  # noqa: F401
finally:
    sys.argv[:] = original_argv
