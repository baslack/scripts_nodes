# scripts_nodes

Small utility nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI).

## Installation

Clone into your ComfyUI `custom_nodes` folder and restart ComfyUI:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/baslack/scripts_nodes.git
```

No extra dependencies are required.

## Nodes

### Enumerator (`utils`)

Turns a selection from a list of strings you define into an integer.

| Input | Type | Description |
|---|---|---|
| `definition` | STRING | Comma-separated entries, e.g. `low, medium, high`. Whitespace is trimmed and empty entries are ignored. |
| `selection` | COMBO | The entry to enumerate. Its options update as you edit `definition`. |
| `first_value` | INT | The value of the first entry. Default `0`. |

| Output | Type | Description |
|---|---|---|
| `value` | INT | `first_value` plus the position of `selection` in `definition`. |

With `definition` = `low, medium, high`, `first_value` = `1`, and `selection` = `high`, the output is `3`.

Notes:

- If `definition` is connected to another node instead of typed in, the dropdown can't update in the editor. The selection is still checked when the workflow runs.
- If the selection isn't in the definition, the node fails with an error listing the valid entries.
- Duplicate entries enumerate to the position of their first occurrence.

### Latent Cache (`latent`)

Saves a latent to disk so the nodes that produce it can be skipped on later runs, including after a restart. Typical use: put it after a first sampling pass and iterate on the refiner without re-running the first pass.

| Input | Type | Description |
|---|---|---|
| `latent` | LATENT | The latent to cache. Only evaluated when the cache needs it. |
| `mode` | COMBO | `auto`, `write`, `read` or `off`. Default `auto`. |
| `name` | STRING | Cache file name, saved as `output/latent_cache/<name>.latent`. Letters, digits, `.`, `_` and `-` only. |

| Output | Type | Description |
|---|---|---|
| `latent` | LATENT | The cached or upstream latent. |

Modes:

- `auto`: loads the cache if it was saved from the same upstream graph, otherwise runs the upstream nodes and saves the result. Changing a seed, prompt, model, LoRA or any other upstream setting causes a re-run.
- `write`: always runs the upstream nodes and saves the result.
- `read`: always loads the cache without running the upstream nodes, even if they have changed since it was saved.
- `off`: passes the latent through without touching the disk.

Notes:

- The upstream nodes are only skipped if nothing else uses their output. Route other consumers, such as preview decodes, through this node's output.
- `auto` sees file names, not file contents. If you replace an input image or model file but keep its name, use `write`.
- A randomized seed changes on every run, so `auto` will always re-run. Fix the seed, or use `read`.
- Audio-video latents (MiniMax H3, LTX 2) are supported.

## License

[MIT](LICENSE)
