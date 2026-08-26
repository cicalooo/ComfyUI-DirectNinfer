import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const MAX_IMAGE_INPUTS = 20;
const IMAGE_SLOT = /^image_(\d+)$/;

function setComboValues(widget, values) {
    if (!widget || !values?.length) return;
    widget.options = widget.options || {};
    const current = widget.value;
    const isPlaceholder =
        typeof current !== "string" || !current || current.startsWith("(no .ninfer");
    let next = values.slice();
    if (!next.includes(current) && !isPlaceholder) {
        next = [current, ...next];
    }
    widget.options.values = next;
    if (!next.includes(widget.value)) {
        widget.value = next[0];
    }
}

function imageInputs(node) {
    return (node.inputs || []).filter(
        (input) => typeof input?.name === "string" && IMAGE_SLOT.test(input.name)
    );
}

function slotIndex(name) {
    const match = IMAGE_SLOT.exec(name);
    return match ? Number.parseInt(match[1], 10) : 0;
}

function syncImageSlots(node) {
    if (!node || node._ampNInferSyncing) return;
    node._ampNInferSyncing = true;
    try {
        let slots = imageInputs(node);
        if (!slots.some((input) => input.name === "image_1")) {
            node.addInput("image_1", "IMAGE");
            slots = imageInputs(node);
        }

        let lastLinked = 0;
        for (const input of slots) {
            if (input.link != null) {
                lastLinked = Math.max(lastLinked, slotIndex(input.name));
            }
        }
        const target = Math.min(MAX_IMAGE_INPUTS, Math.max(1, lastLinked + 1));
        const present = new Set(slots.map((input) => input.name));

        for (let index = 1; index <= target; index++) {
            const name = `image_${index}`;
            if (!present.has(name)) {
                node.addInput(name, "IMAGE");
                present.add(name);
            }
        }

        slots = imageInputs(node)
            .map((input) => ({ input, index: slotIndex(input.name) }))
            .filter((entry) => entry.index > target)
            .sort((a, b) => b.index - a.index);
        for (const { input } of slots) {
            if (input.link != null) continue;
            const position = node.inputs.indexOf(input);
            if (position >= 0) {
                node.removeInput(position);
            }
        }

        if (typeof node.setSize === "function" && typeof node.computeSize === "function") {
            node.setSize(node.computeSize());
        }
        node.setDirtyCanvas?.(true, true);
    } finally {
        node._ampNInferSyncing = false;
    }
}

app.registerExtension({
    name: "ampNInfer.ui",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "NInferQwenNode") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            const modelsDirWidget = this.widgets?.find((w) => w.name === "models_dir");
            const modelWidget = this.widgets?.find((w) => w.name === "model_artifact");

            const refresh = async () => {
                const dir = modelsDirWidget ? modelsDirWidget.value : "";
                try {
                    const resp = await api.fetchApi("/amp_ninfer/scan_models", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ dir }),
                    });
                    const data = await resp.json();
                    if (data.error) {
                        console.warn("[Amp NInfer]", data.error);
                    }
                    setComboValues(modelWidget, data.models);
                    this.setDirtyCanvas(true, true);
                } catch (error) {
                    console.error("[Amp NInfer] Failed to refresh models:", error);
                }
            };

            this.addWidget("button", "Refresh", null, refresh);
            refresh();
            syncImageSlots(this);
            return result;
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            const result = onConnectionsChange
                ? onConnectionsChange.apply(this, arguments)
                : undefined;
            syncImageSlots(this);
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure ? onConfigure.apply(this, arguments) : undefined;
            syncImageSlots(this);
            return result;
        };
    },
});
