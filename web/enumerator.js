import { app } from "../../scripts/app.js";

/**
 * Split an enumerator definition into its entries. Must match parse_definition in enumerator.py.
 * @param {string} definition Comma-separated entries.
 * @returns {string[]} The trimmed, non-empty entries in definition order.
 */
function parseDefinition(definition) {
    return definition.split(",").map((s) => s.trim()).filter(Boolean);
}

/**
 * Repopulate the selection combo from the node's definition widget.
 * @param {LGraphNode} node An Enumerator node.
 * @returns {void}
 */
function updateSelection(node) {
    const definition = node.widgets.find((w) => w.name === "definition");
    const selection = node.widgets.find((w) => w.name === "selection");
    const options = parseDefinition(definition.value);
    selection.options.values = options;
    // Keep the saved selection while a workflow is loading; the definition may not be applied yet.
    if (!app.configuringGraph && !options.includes(selection.value)) {
        selection.value = options[0] ?? "";
    }
}

app.registerExtension({
    name: "scripts_nodes.Enumerator",
    /**
     * @param {typeof LGraphNode} nodeType
     * @param {ComfyNodeDef} nodeData
     * @returns {void}
     */
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "Enumerator") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            const definition = this.widgets.find((w) => w.name === "definition");
            const callback = definition.callback;
            definition.callback = (...args) => {
                callback?.apply(definition, args);
                updateSelection(this);
            };
            updateSelection(this);
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            onConfigure?.apply(this, arguments);
            updateSelection(this);
        };
    },
});
