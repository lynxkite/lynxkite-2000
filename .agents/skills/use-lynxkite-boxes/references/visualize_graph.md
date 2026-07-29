**Visualize graph:**
Visualizes the graph using ECharts and allows the user to customize the visualization through "chips".
```python
@op("Visualize graph", view="graph_visualization", icon="eye", color="blue")
def visualize_graph(b: core.Bundle, *, chip_data: str = ""):
    """
    Visualizes the graph using ECharts and allows the user to customize the visualization through "chips".
    :param b: the bundle
    :param chip_data: the frontend uses this parameter to store relevant data of the chips
    """

    b = b.copy()
    (nodes, node_id), (edges, source_id, target_id) = _nodes_and_edges(b)

    item_count = len(nodes) + len(edges)
    if item_count < 10000:
        pos = nx.spring_layout(b.to_nx(), iterations=max(1, int(10000 / item_count)))
    else:
        pos = {node_id: np.random.rand(2) for node_id in nodes[node_id]}

    edges = bundle.df_for_frontend(edges, 10000)
    nodes = bundle.df_for_frontend(nodes, 10000)

    node_columns = [col for col in nodes.columns]
    edge_columns = [col for col in edges.columns]

    nodes_dict = nodes.to_dict(orient="index")
    edges = edges.to_records()

    v = {
        "animationDuration": 500,
        "animationEasingUpdate": "quinticInOut",
        "tooltip": {"show": True},
        "series": [
            {
                "type": "graph",
                "lineStyle": {
                    "color": "gray",
                    "curveness": 0.3,
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
                        "id": str(node_id),
                        "x": float(pos[node_id][0]),
                        "y": float(pos[node_id][1]),
                        "symbolSize": 50 / len(nodes) ** 0.5,
                        "attributes": {col: str(record[col]) for col in node_columns},
                    }
                    for node_id, record in nodes_dict.items()
                ],
                "links": [
                    {
                        "source": str(getattr(r, source_id, "")),
                        "target": str(getattr(r, target_id, "")),
                        "attributes": {col: str(getattr(r, col)) for col in edge_columns},
                    }
                    for r in edges
                ],
            },
        ],
    }
    return v

```
