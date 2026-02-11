"""Visualizations."""

from lynxkite_core import ops

from .. import core
import matplotlib.cm
import matplotlib.colors
import networkx as nx
import pandas as pd


op = ops.op_registration(core.ENV)


def _map_color(value):
    if pd.api.types.is_numeric_dtype(value):
        cmap = matplotlib.cm.get_cmap("viridis")
        value = (value - value.min()) / (value.max() - value.min())
        rgba = cmap(value.values)
        return [
            "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
            for r, g, b in rgba[:, :3]
        ]
    else:
        cmap = matplotlib.cm.get_cmap("Paired")
        categories = {k: i for i, k in enumerate(value.unique())}
        assert isinstance(cmap, matplotlib.colors.ListedColormap)
        colors = list(
            cmap.colors[: len(categories)]  # ty: ignore[not-subscriptable, invalid-argument-type]
        )
        return [
            "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
            for r, g, b in [colors[min(len(colors) - 1, categories[v])] for v in value]
        ]


@op("Visualize graph", view="visualization", icon="eye", color="blue")
def visualize_graph(
    graph: core.Bundle,
    *,
    color_nodes_by: core.NodePropertyName | None = None,
    label_by: core.NodePropertyName | None = None,
    color_edges_by: core.EdgePropertyName | None = None,
):
    nodes = core.df_for_frontend(graph.dfs["nodes"], 10_000)
    if color_nodes_by:
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
            scale_y = scale_x
            if y == "lat":
                scale_y *= -1
            pos = {
                node_id: ((row[x] - cx) * scale_x, (row[y] - cy) * scale_y)
                for node_id, row in nodes.iterrows()
            }
            curveness = 0  # Street maps are better with straight streets.
            break
    else:
        pos = nx.spring_layout(graph.to_nx(), iterations=max(1, int(10000 / len(nodes))))
        curveness = 0.3
    nodes = nodes.to_records()
    deduped_edges = graph.dfs["edges"].drop_duplicates(["source", "target"])
    edges = core.df_for_frontend(deduped_edges, 10_000)
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
                        "id": str(n.id),
                        "x": float(pos[n.id][0]),
                        "y": float(pos[n.id][1]),
                        # Adjust node size to cover the same area no matter how many nodes there are.
                        "symbolSize": 50 / len(nodes) ** 0.5,
                        "itemStyle": {"color": n.color} if color_nodes_by else {},
                        "label": {"show": label_by is not None},
                        "name": format_label(getattr(n, label_by, "")) if label_by else None,
                        "value": str(getattr(n, color_nodes_by, "")) if color_nodes_by else None,
                    }
                    for n in nodes
                ],
                "links": [
                    {
                        "source": str(r.source),
                        "target": str(r.target),
                        "lineStyle": {"color": r.color} if color_edges_by else {},
                        "value": str(getattr(r, color_edges_by, "")) if color_edges_by else None,
                    }
                    for r in edges
                ],
            },
        ],
    }
    return v
