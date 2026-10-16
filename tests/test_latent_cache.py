"""Contract tests for Latent Cache, with and without a latent connected.

A node whose upstream sampler is muted arrives without its ``latent`` input:
the editor drops links from muted nodes when it queues the prompt.
"""
import json

import pytest
import torch

import folder_paths

import scripts_nodes
from scripts_nodes import latent_cache


@pytest.fixture(autouse=True)
def output_dir(tmp_path):
    original = folder_paths.get_output_directory()
    folder_paths.set_output_directory(str(tmp_path))
    yield tmp_path
    folder_paths.set_output_directory(original)


def _node():
    return scripts_nodes.NODE_CLASS_MAPPINGS["LatentCache"]()


def _prompt(mode, seed=1, connected=True):
    inputs = {"mode": mode, "name": "unit_2"}
    if connected:
        inputs["latent"] = ["5", 0]
    return {
        "5": {"class_type": "KSampler", "inputs": {"seed": seed}},
        "9": {"class_type": "LatentCache", "inputs": inputs},
    }


def _run(mode, latent=None, seed=1, connected=True):
    """Run the node the way the executor does: ask which lazy inputs it needs, then call it."""
    node, prompt = _node(), _prompt(mode, seed, connected)
    needed = node.check_lazy_status(mode=mode, name="unit_2", prompt=prompt, unique_id="9")
    given = {"latent": latent} if "latent" in needed else {}
    return needed, node.cache(mode=mode, name="unit_2", prompt=prompt, unique_id="9", **given)[0]


def _latent(value):
    return {"samples": torch.full((1, 4, 2, 2), float(value))}


def test_latent_is_an_optional_lazy_input():
    inputs = scripts_nodes.NODE_CLASS_MAPPINGS["LatentCache"].INPUT_TYPES()
    assert "latent" not in inputs["required"]
    assert inputs["optional"]["latent"][1]["lazy"] is True


@pytest.mark.parametrize("mode", ["auto", "read"])
def test_with_nothing_connected_the_saved_latent_is_loaded(mode):
    _run("auto", _latent(3))
    needed, out = _run(mode, connected=False)
    assert needed == []
    torch.testing.assert_close(out["samples"], _latent(3)["samples"])


def test_auto_with_nothing_connected_loads_even_a_cache_from_another_graph():
    _run("auto", _latent(3), seed=1)
    _, out = _run("auto", seed=2, connected=False)
    torch.testing.assert_close(out["samples"], _latent(3)["samples"])


@pytest.mark.parametrize("mode", ["auto", "write", "off"])
def test_with_nothing_connected_and_nothing_saved_it_fails_clearly(mode):
    with pytest.raises(ValueError, match="'unit_2' has no latent connected"):
        _run(mode, connected=False)


def test_connected_auto_still_saves_then_reuses_and_resamples_on_change():
    needed, out = _run("auto", _latent(3))
    assert needed == ["latent"]
    needed, out = _run("auto", _latent(4))
    assert needed == []
    torch.testing.assert_close(out["samples"], _latent(3)["samples"])
    needed, out = _run("auto", _latent(5), seed=2)
    assert needed == ["latent"]
    torch.testing.assert_close(out["samples"], _latent(5)["samples"])


def test_directory_is_an_optional_socket():
    inputs = scripts_nodes.NODE_CLASS_MAPPINGS["LatentCache"].INPUT_TYPES()
    assert inputs["optional"]["directory"][1]["forceInput"] is True
    assert not inputs["optional"]["directory"][1].get("lazy")


def _run_in(directory, mode, latent=None):
    node, prompt = _node(), _prompt(mode)
    prompt["9"]["inputs"]["directory"] = directory
    needed = node.check_lazy_status(mode=mode, name="unit_2", prompt=prompt, unique_id="9", directory=directory)
    given = {"latent": latent} if "latent" in needed else {}
    return needed, node.cache(mode=mode, name="unit_2", prompt=prompt, unique_id="9", directory=directory, **given)[0]


def test_a_cache_kept_in_a_user_directory_folder_is_reused(tmp_path):
    user = tmp_path / "user"
    original = folder_paths.get_user_directory()
    folder_paths.set_user_directory(str(user))
    try:
        folder = user / "h3_extender_cache" / "chain_cabin.latents"
        _run_in(str(folder), "auto", _latent(3))
        assert (folder / "unit_2.latent").is_file()
        needed, out = _run_in(str(folder), "auto")
        assert needed == []
        torch.testing.assert_close(out["samples"], _latent(3)["samples"])
    finally:
        folder_paths.set_user_directory(original)


def test_a_directory_outside_comfyui_is_refused(tmp_path):
    with pytest.raises(ValueError, match="outside ComfyUI's user and output directories"):
        _run_in(str(tmp_path.parent / "elsewhere"), "auto", _latent(3))


def test_moving_a_cache_to_another_folder_keeps_the_caches_after_it(output_dir):
    first = {"5": {"class_type": "KSampler", "inputs": {"seed": 1}},
             "7": {"class_type": "LatentCache", "inputs": {"latent": ["5", 0], "mode": "auto", "name": "unit_1"}},
             "9": {"class_type": "LatentCache", "inputs": {"latent": ["7", 0], "mode": "auto", "name": "unit_2"}}}
    moved = json.loads(json.dumps(first))
    moved["7"]["inputs"]["directory"] = str(output_dir / "project_a")
    renamed = json.loads(json.dumps(first))
    renamed["7"]["inputs"]["name"] = "unit_1b"
    hash_of = latent_cache.upstream_hash
    assert hash_of(moved, "9") == hash_of(first, "9")
    assert hash_of(renamed, "9") != hash_of(first, "9")
