**Visualize graph:**

```python
@op("Visualize graph", view="visualization", icon="eye", color="blue")
def visualize_graph(
    graph: core.Bundle,
    *,
    color_nodes_by: core.NodePropertyName | None = None,
    label_by: core.NodePropertyName | None = None,
    color_edges_by: core.EdgePropertyName | None = None,
):
    graph = graph.copy()
    (nodes, node_id), (edges, source_id, target_id) = _nodes_and_edges(graph)
    if color_nodes_by and color_nodes_by in nodes.columns:
        nodes["color"] = _map_color(nodes[color_nodes_by])

    for cols in ["x y", "long lat"]:
        x, y = cols.split()
        if (
            x in nodes.columns
            and nodes[x].dtype == "float64"
            and y in nodes.columns
            and nodes[y].dtype == "float64"
        ):
            cx, cy = nodes[x].mean(), nodes[y].mean()
            dx, dy = nodes[x].std(), nodes[y].std()
            # Scale up to avoid float precision issues and because eCharts omits short edges.
            scale_x = 100 / max(dx, dy)
            scale_y = -scale_x
            pos = {
                node_id: ((row[x] - cx) * scale_x, (row[y] - cy) * scale_y)
                for node_id, row in nodes.iterrows()
            }
            curveness = 0  # Street maps are better with straight streets.
            break
    else:
        pos = nx.spring_layout(graph.to_nx(), iterations=max(1, int(10000 / len(nodes))))
        curveness = 0.3
    nodes = nodes.to_records(index=False)
    if color_edges_by:
        edges["color"] = _map_color(edges[color_edges_by])
    edges = edges.to_records()

    def format_label(value):
        if pd.isna(value):
            return ""
        elif isinstance(value, float):
            return f"{value:.2f}"
        else:
            return str(value)

    v = {
        "animationDuration": 500,
        "animationEasingUpdate": "quinticInOut",
        "tooltip": {"show": True},
        "series": [
            {
                "type": "graph",
                # Mouse zoom/panning is disabled for now. It interacts badly with ReactFlow.
                # "roam": True,
                "lineStyle": {
                    "color": "gray",
                    "curveness": curveness,
                },
                "emphasis": {
                    "focus": "adjacency",
                    "lineStyle": {
                        "width": 10,
                    },
                },
                "label": {"position": "top", "formatter": "{b}"},
                "data": [
                    {
                        "id": str(n[node_id]),
                        "x": float(pos[n[node_id]][0]),
                        "y": float(pos[n[node_id]][1]),
                        # Adjust node size to cover the same area no matter how many nodes there are.
                        "symbolSize": 50 / len(nodes) ** 0.5,
                        "itemStyle": {"color": getattr(n, "color", None)} if color_nodes_by else {},
                        "label": {"show": label_by is not None},
                        "name": format_label(getattr(n, label_by, "")) if label_by else None,
                        "value": str(getattr(n, color_nodes_by, "")) if color_nodes_by else None,
                    }
                    for n in nodes
                ],
                "links": [
                    {
                        "source": str(getattr(r, source_id, "")),
                        "target": str(getattr(r, target_id, "")),
                        "lineStyle": {"color": getattr(r, "color", None)} if color_edges_by else {},
                        "value": str(getattr(r, color_edges_by, "")) if color_edges_by else None,
                    }
                    for r in edges
                ],
            },
        ],
    }
    return v

```
Custom types:
  - color_nodes_by: typing.Optional[typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].nodes[].columns[]'}]]
  - label_by: typing.Optional[typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].nodes[].columns[]'}]]
  - color_edges_by: typing.Optional[typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].edges[].columns[]'}]]
