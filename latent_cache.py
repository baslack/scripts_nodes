import hashlib
import json
import logging
import os
import re

import safetensors
import safetensors.torch
import torch

import folder_paths
import nodes
from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict
from comfy.nested_tensor import NestedTensor
from comfy_execution.graph_utils import is_link

FORMAT = "scripts_nodes.latent_cache/1"
NAME_PATTERN = re.compile(r"[A-Za-z0-9._-]+")
# An input option that marks an input as naming only, such as a project name: the upstream hash skips it.
IGNORE_OPTION = "latent_cache_ignore"


def _inside(root: str, path: str) -> bool:
    try:
        return os.path.commonpath([root, path]) == root
    except ValueError:  # on another drive
        return False


def cache_path(name: str, directory: str | None = None) -> str:
    """Resolve a cache name to its file.

    Args:
        name: The cache name. Letters, digits, ``.``, ``_`` and ``-`` only.
        directory: The folder to keep it in, inside ComfyUI's user or output
            directory. ``output/latent_cache`` when not given.

    Returns:
        The absolute path of ``<name>.latent``.

    Raises:
        ValueError: If ``name`` contains other characters, or the file would
            be outside the user and output directories.
    """
    if not NAME_PATTERN.fullmatch(name):
        raise ValueError(f"LatentCache: name '{name}' may only contain letters, digits, '.', '_' and '-'")
    if directory:
        cache_dir = os.path.realpath(directory)
        roots = [os.path.realpath(folder_paths.get_user_directory()), os.path.realpath(folder_paths.get_output_directory())]
        if not any(_inside(root, cache_dir) for root in roots):
            raise ValueError(f"LatentCache: directory '{directory}' is outside ComfyUI's user and output directories")
    else:
        cache_dir = os.path.abspath(os.path.join(folder_paths.get_output_directory(), "latent_cache"))
    path = os.path.abspath(os.path.join(cache_dir, f"{name}.latent"))
    if os.path.commonpath([cache_dir, path]) != cache_dir:
        raise ValueError(f"LatentCache: name '{name}' resolves outside {cache_dir}")
    return path


def _ignored_inputs(class_type: str, legacy: bool) -> frozenset:
    """The inputs of a node class the upstream hash skips."""
    if legacy:
        return frozenset({"directory"}) if class_type == "LatentCache" else frozenset()
    node_class = NODE_CLASS_MAPPINGS.get(class_type) or nodes.NODE_CLASS_MAPPINGS.get(class_type)
    if node_class is None:
        return frozenset()
    spec = node_class.INPUT_TYPES()
    return frozenset(
        name for group in ("required", "optional") for name, value in (spec.get(group) or {}).items()
        if len(value) > 1 and isinstance(value[1], dict) and value[1].get(IGNORE_OPTION)
    )


def upstream_hash(prompt: dict, node_id: str, input_name: str = "latent", legacy: bool = False) -> str:
    """Hash everything that feeds one input of a node.

    Each upstream node is hashed from its class and literal inputs, with links
    replaced by the hash of the node they come from, so the result does not
    depend on node ids. Inputs whose options set ``latent_cache_ignore`` are
    left out, so naming things (a project, a cache's folder) doesn't change
    the hash.

    Args:
        prompt: The API-format prompt.
        node_id: The node whose input is hashed.
        input_name: The input to hash.
        legacy: Hash the way caches saved before ``latent_cache_ignore`` were
            hashed, skipping only a Latent Cache's ``directory``.

    Returns:
        A hex SHA-256 digest.
    """
    memo = {}
    ignored = {}

    def value_key(value):
        if is_link(value):
            return ["link", node_key(value[0]), value[1]]
        return value

    def node_key(nid):
        if nid not in memo:
            node = prompt[nid]
            if node["class_type"] not in ignored:
                ignored[node["class_type"]] = _ignored_inputs(node["class_type"], legacy)
            skip = ignored[node["class_type"]]
            inputs = {k: value_key(v) for k, v in node["inputs"].items() if k not in skip}
            data = json.dumps([node["class_type"], inputs], sort_keys=True)
            memo[nid] = hashlib.sha256(data.encode("utf-8")).hexdigest()
        return memo[nid]

    data = json.dumps(value_key(prompt[node_id]["inputs"][input_name]), sort_keys=True)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def save_latent(path: str, latent: dict, upstream: str) -> None:
    """Write a latent dict to a safetensors file.

    Tensors are stored under their key and :class:`NestedTensor` parts under
    ``<key>.<index>``. Other values are stored as JSON in the metadata. The
    file is written beside ``path`` and renamed into place, so an interrupted
    save never leaves a truncated cache.

    Args:
        path: The destination file.
        latent: The latent dict.
        upstream: The upstream hash to store with it.

    Raises:
        TypeError: If a value is not a tensor, a NestedTensor, or JSON data.
    """
    tensors = {}
    layout = {}
    values = {}
    for key, value in latent.items():
        if isinstance(value, NestedTensor):
            layout[key] = len(value.tensors)
            for i, t in enumerate(value.tensors):
                tensors[f"{key}.{i}"] = t.contiguous().cpu()
        elif isinstance(value, torch.Tensor):
            layout[key] = None
            tensors[key] = value.contiguous().cpu()
        else:
            try:
                json.dumps(value)
            except TypeError:
                raise TypeError(f"LatentCache: can't store latent['{key}'] of type {type(value).__name__}") from None
            values[key] = value

    metadata = {
        "format": FORMAT,
        "upstream": upstream,
        "layout": json.dumps(layout),
        "values": json.dumps(values),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    safetensors.torch.save_file(tensors, tmp_path, metadata=metadata)
    os.replace(tmp_path, path)


def read_metadata(path: str) -> dict | None:
    """Read a cache file's metadata.

    Args:
        path: The cache file.

    Returns:
        The metadata, or ``None`` if the file does not exist.

    Raises:
        ValueError: If the file is not a latent cache.
    """
    if not os.path.isfile(path):
        return None
    with safetensors.safe_open(path, framework="pt") as f:
        metadata = f.metadata() or {}
    if metadata.get("format") != FORMAT:
        raise ValueError(f"LatentCache: {path} is not a latent cache file")
    return metadata


def load_latent(path: str) -> dict:
    """Read a latent dict written by :func:`save_latent`.

    Args:
        path: The cache file.

    Returns:
        The latent dict, with tensors on the CPU.
    """
    metadata = read_metadata(path)
    if metadata is None:
        raise FileNotFoundError(f"LatentCache: no cache at {path}")
    latent = json.loads(metadata["values"])
    with safetensors.safe_open(path, framework="pt", device="cpu") as f:
        for key, count in json.loads(metadata["layout"]).items():
            if count is None:
                latent[key] = f.get_tensor(key)
            else:
                latent[key] = NestedTensor([f.get_tensor(f"{key}.{i}") for i in range(count)])
    return latent


class LatentCache(ComfyNodeABC):
    """Keep a latent on disk so the nodes that produce it can be skipped.

    The ``latent`` input is lazy: on a cache hit it is never evaluated, so the
    upstream sampler does not run, as long as nothing else needs its output.
    It is also optional, so muting the sampler leaves a node that still loads
    the saved latent when something asks for it.
    """

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        """Declare the node inputs.

        Returns:
            The ``mode`` combo, the cache ``name``, the optional lazy
            ``latent`` and ``directory``, and the hidden prompt and node id
            used to hash the upstream graph.
        """
        return {
            "required": {
                "mode": (["auto", "write", "read", "off"], {"default": "auto"}),
                "name": (IO.STRING, {"default": "latent_cache"}),
            },
            "optional": {
                "latent": (IO.LATENT, {"lazy": True}),
                "directory": (IO.STRING, {"forceInput": True, IGNORE_OPTION: True, "tooltip": "Where to keep the cache, inside ComfyUI's user or output directory, e.g. a project's latent_directory. Default: output/latent_cache."}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = (IO.LATENT,)
    RETURN_NAMES = ("latent",)
    FUNCTION = "cache"
    CATEGORY = "latent"

    def is_hit(self, mode: str, name: str, prompt: dict, unique_id: str, directory: str | None = None) -> bool:
        """Decide whether the cache answers without evaluating ``latent``.

        Args:
            mode: ``auto``, ``write``, ``read`` or ``off``.
            name: The cache name.
            prompt: The API-format prompt.
            unique_id: This node's id.
            directory: The cache's folder, if not the default.

        Returns:
            ``True`` for ``read``, and for ``auto`` when a cache exists and
            either nothing is connected to ``latent`` or the stored upstream
            hash matches the current graph.
        """
        if mode == "read":
            return True
        if mode != "auto":
            return False
        metadata = read_metadata(cache_path(name, directory))
        if metadata is None:
            return False
        if "latent" not in prompt[unique_id]["inputs"]:
            return True
        return metadata["upstream"] in (upstream_hash(prompt, unique_id), upstream_hash(prompt, unique_id, legacy=True))

    def check_lazy_status(self, mode: str, name: str, prompt: dict, unique_id: str, latent: dict | None = None,
                          directory: str | None = None) -> list[str]:
        """Request ``latent`` only on a cache miss.

        Args:
            mode: The cache mode.
            name: The cache name.
            prompt: The API-format prompt.
            unique_id: This node's id.
            latent: The upstream latent, if already evaluated.
            directory: The cache's folder, if not the default.

        Returns:
            ``[]`` on a hit or when nothing is connected, otherwise ``["latent"]``.
        """
        if "latent" not in prompt[unique_id]["inputs"]:
            return []
        return [] if self.is_hit(mode, name, prompt, unique_id, directory) else ["latent"]

    def cache(self, mode: str, name: str, prompt: dict, unique_id: str, latent: dict | None = None,
              directory: str | None = None) -> tuple[dict]:
        """Load, save, or pass through the latent according to ``mode``.

        Args:
            mode: ``off`` passes through, ``write`` always saves, ``read``
                always loads, and ``auto`` loads when the stored upstream hash
                matches and saves otherwise.
            name: The cache name.
            prompt: The API-format prompt.
            unique_id: This node's id.
            latent: The upstream latent, or ``None`` on a cache hit.
            directory: The cache's folder, if not the default.

        Returns:
            The cached or upstream latent.

        Raises:
            ValueError: If the latent is needed but nothing is connected.
        """
        hit = mode != "off" and self.is_hit(mode, name, prompt, unique_id, directory)
        if not hit and latent is None:
            raise ValueError(f"LatentCache: '{name}' has no latent connected and no saved cache to load in mode '{mode}'.")
        if mode == "off":
            return (latent,)
        path = cache_path(name, directory)
        if hit:
            loaded = load_latent(path)
            # TODO: drop this upgrade, and legacy=True, once caches saved before latent_cache_ignore are gone.
            if "latent" in prompt[unique_id]["inputs"]:
                current = upstream_hash(prompt, unique_id)
                if read_metadata(path)["upstream"] == upstream_hash(prompt, unique_id, legacy=True) != current:
                    save_latent(path, loaded, current)
                    logging.info(f"LatentCache: upgraded the stored hash of '{name}'")
            logging.info(f"LatentCache: loaded '{name}'")
            return (loaded,)
        save_latent(path, latent, upstream_hash(prompt, unique_id))
        logging.info(f"LatentCache: saved '{name}'")
        return (latent,)


NODE_CLASS_MAPPINGS: dict[str, type[ComfyNodeABC]] = {
    "LatentCache": LatentCache,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "LatentCache": "Latent Cache",
}
