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

### Remove Preview Override (`utils`)

Removes a named hook from a model, by default the live preview that [KJNodes](https://github.com/kijai/ComfyUI-KJNodes)' Model Preview Override attaches. Typical use: preview the first sampling pass but not the refine pass that uses the same model.

| Input | Type | Description |
|---|---|---|
| `model` | MODEL | The model. |
| `name` | STRING | The name the hook was registered under. Default `kj_preview_override`. |

| Output | Type | Description |
|---|---|---|
| `model` | MODEL | A copy of the model without that hook. |

Notes:

- The original model keeps the hook, so samplers that use it are still previewed.
- A model without the hook, for example with the preview node bypassed, passes through unchanged.
- `kj_preview_override` is KJNodes' internal name. If a KJNodes update renames it, the preview comes back on the refine pass; put the new name in `name`.

### H3 LoRA on Audio or Video Tokens (`loaders`)

Applies a LoRA to only the audio tokens, or only the video tokens, of a MiniMax H3 model. H3 runs video and audio through the same transformer weights, so a normal LoRA changes both; use this one to take, say, a voice LoRA's sound without its effect on the picture.

| Input | Type | Description |
|---|---|---|
| `model` | MODEL | A MiniMax H3 model. |
| `lora_name` | COMBO | A LoRA in `models/loras`. |
| `strength` | FLOAT | Multiplier on the LoRA's change. Default `1.0`. |
| `tokens` | COMBO | `audio` or `video`: the stream the LoRA changes. Default `audio`. |

| Output | Type | Description |
|---|---|---|
| `model` | MODEL | A copy of the model with the LoRA limited to those tokens. |

Notes:

- The weights are left alone. While each transformer block runs, the LoRA's change is added only to the chosen stream's rows of each LoRA'd layer. Reference and keyframe rows count with their stream.
- The other stream still reacts: H3's video attends to its audio, so changed audio still shapes the video, as it does for lip sync. Only the weights it sees stay untouched.
- Layers outside the transformer blocks, such as a LoRA's text-refiner layers, are left out; the log says how many.
- Plain LoRA only: LoCon mid weights, DoRA, LoHa and LoKr are refused. It chains with other nodes' block patches and with ordinary LoRA loaders.

## Tests

Run with ComfyUI's own Python; `pytest` is the only extra package. The GPU is not used.

```powershell
cd <this repo>
$env:COMFYUI_PATH = "<ComfyUI checkout>"
$env:CUDA_VISIBLE_DEVICES = "-1"
<ComfyUI python> -m pytest tests
```

## License

[MIT](LICENSE)
