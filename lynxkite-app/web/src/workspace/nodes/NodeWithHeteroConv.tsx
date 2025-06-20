import { useReactFlow } from "@xyflow/react";
import React from "react";
// @ts-ignore
import Triangle from "~icons/tabler/triangle-inverted-filled.jsx";
import LynxKiteNode from "./LynxKiteNode";

function parseLayers(raw: any): any[] {
  try {
    return JSON.parse(raw || "[]");
  } catch {
    return [];
  }
}

export default LynxKiteNode(function NodeWithHeteroConv(props: any) {
  const reactFlow = useReactFlow();
  const [collapsed, setCollapsed] = React.useState(props.collapsed);
  const [layers, setLayers] = React.useState(() => parseLayers(props.data.params.layers));

  function updateLayers(newLayers: any[]) {
    setLayers(newLayers);
    reactFlow.updateNodeData(props.id, {
      params: { ...props.data.params, layers: JSON.stringify(newLayers) },
    });
  }

  function updateLayer(idx: number, field: string, value: any) {
    const nl = layers.slice();
    const layer = { ...nl[idx] };
    if (field.startsWith("relation")) {
      const i = Number.parseInt(field.replace("relation", ""));
      const r = [...(layer.relation || ["", "", ""])] as string[];
      r[i] = value;
      layer.relation = r;
    } else if (field === "type") {
      layer.type = value;
    } else {
      layer.params = { ...(layer.params || {}), [field]: value };
    }
    nl[idx] = layer;
    updateLayers(nl);
  }

  function addLayer() {
    updateLayers([
      ...layers,
      {
        relation: ["src", "rel", "dst"],
        type: "GraphConv",
        params: { in_channels: 4, out_channels: 4 },
      },
    ]);
  }

  function removeLayer(idx: number) {
    const nl = layers.slice();
    nl.splice(idx, 1);
    updateLayers(nl);
  }

  return (
    <>
      {props.collapsed && (
        <div className="params-expander" onClick={() => setCollapsed(!collapsed)}>
          <Triangle className={`flippy ${collapsed ? "flippy-90" : ""}`} />
        </div>
      )}
      {!collapsed && (
        <div className="graph-relations">
          <div className="graph-table-header">
            Convolutions
            <button className="add-relationship-button" onClick={() => addLayer()}>
              +
            </button>
          </div>
          {layers.map((layer, idx) => (
            <div key={idx} className="graph-relation-attributes">
              <label>Relation</label>
              <div className="flex gap-1">
                <input
                  value={layer.relation?.[0] || ""}
                  onChange={(e) => updateLayer(idx, "relation0", e.currentTarget.value)}
                />
                <input
                  value={layer.relation?.[1] || ""}
                  onChange={(e) => updateLayer(idx, "relation1", e.currentTarget.value)}
                />
                <input
                  value={layer.relation?.[2] || ""}
                  onChange={(e) => updateLayer(idx, "relation2", e.currentTarget.value)}
                />
              </div>
              <label>Type</label>
              <select
                value={layer.type}
                onChange={(e) => updateLayer(idx, "type", e.currentTarget.value)}
              >
                <option value="GraphConv">GraphConv</option>
                <option value="GATConv">GATConv</option>
              </select>
              <label>in_channels</label>
              <input
                type="number"
                value={layer.params?.in_channels ?? 0}
                onChange={(e) => updateLayer(idx, "in_channels", Number(e.currentTarget.value))}
              />
              <label>out_channels</label>
              <input
                type="number"
                value={layer.params?.out_channels ?? 0}
                onChange={(e) => updateLayer(idx, "out_channels", Number(e.currentTarget.value))}
              />
              {layer.type === "GATConv" && (
                <>
                  <label>heads</label>
                  <input
                    type="number"
                    value={layer.params?.heads ?? 1}
                    onChange={(e) => updateLayer(idx, "heads", Number(e.currentTarget.value))}
                  />
                </>
              )}
              <button onClick={() => removeLayer(idx)}>x</button>
            </div>
          ))}
        </div>
      )}
    </>
  );
});
