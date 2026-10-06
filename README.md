# scripts_nodes

Small utility nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI).

## Installation

Clone into your ComfyUI `custom_nodes` folder and restart ComfyUI:

```bash
cd ComfyUI/custom_nodes
git clone <repo-url> scripts_nodes
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

## License

[MIT](LICENSE)
