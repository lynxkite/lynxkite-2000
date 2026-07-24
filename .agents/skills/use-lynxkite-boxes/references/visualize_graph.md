**Visualize graph:**

```python
@op("Visualize graph", view="visualization", icon="eye", color="blue")
def visualize_graph(b: core.Bundle):
    b = b.copy()
    (nodes, node_id), (edges_df, source_id, target_id) = _nodes_and_edges(b)

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
            scale_x = 100 / max(dx, dy)
            scale_y = -scale_x
            pos = {
                node_id: ((row[x] - cx) * scale_x, (row[y] - cy) * scale_y)
                for node_id, row in nodes.iterrows()
            }
            curveness = 0
            break
    else:
        pos = nx.spring_layout(b.to_nx(), iterations=max(1, int(10000 / len(nodes))))
        curveness = 0.3

    node_columns = [col for col in nodes.columns]
    edge_columns = [col for col in edges_df.columns]

    nodes_dict = nodes.to_dict(orient="index")
    edges = edges_df.to_records()

    v = {
        "animationDuration": 500,
        "animationEasingUpdate": "quinticInOut",
        "tooltip": {"show": True},
        "series": [
            {
                "type": "graph",
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
                        "id": str(node_id),
                        "x": float(pos[node_id][0]),
                        "y": float(pos[node_id][1]),
                        "symbolSize": 50 / len(nodes) ** 0.5,
                        "attributes": {
                            col: str(record[col]) for col in node_columns if pd.notna(record[col])
                        },
                    }
                    for node_id, record in nodes_dict.items()
                ],
                "links": [
                    {
                        "source": str(getattr(r, source_id, "")),
                        "target": str(getattr(r, target_id, "")),
                        "attributes": {
                            col: str(getattr(r, col))
                            for col in edge_columns
                            if pd.notna(getattr(r, col))
                        },
                    }
                    for r in edges
                ],
            },
        ],
    }
    return v

```
