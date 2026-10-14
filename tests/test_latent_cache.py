"""Contract tests for Latent Cache, with and without a latent connected.

A node whose upstream sampler is muted arrives without its ``latent`` input:
the editor drops links from muted nodes when it queues the prompt.
"""
import pytest
import torch

import folder_paths

import scripts_nodes


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
