"""Visualizations."""

from lynxkite_core import ops

from lynxkite_graph_analytics import core
import matplotlib.cm
import matplotlib.colors
import networkx as nx
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


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
        value = value.fillna("NaN").astype(str)
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
    edges_df = graph.dfs["edges"]

    sources = set(edges_df["source"])
    targets = set(edges_df["target"])

    def get_role(node_id):
        is_source = node_id in sources
        is_target = node_id in targets
        if is_source and is_target:
            return "both"
        elif is_source:
            return "source"
        elif is_target:
            return "target"
        else:
            return "isolated"

    nodes["role"] = [get_role(nid) for nid in nodes.index]

    color_map = {
        "source": "#5eabe2",   # blue
        "target": "#c25a5a",   # red
        "both": "#6abf6a",     # green
        "isolated": "#aaaaaa", # gray
    }

    nodes["color"] = nodes["role"].map(color_map)
    for cols in []:
        x, y = cols.split()
        if (
            x in nodes.columns
            and nodes[x].dtype == "float64"
            and y in nodes.columns
            and nodes[y].dtype == "float64"
        ):
            cx, cy = nodes[x].mean(), nodes[y].mean()
            dx, dy = nodes[x].std(), nodes[y].std()
            # Scale up to avoid float precision issues.
            scale_x = 100 / max(dx, dy)
            scale_y = scale_x
            if y == "lat":
                scale_y *= -1
            pos = {
                node_id: ((row[x] - cx) * scale_x, (row[y] - cy) * scale_y)
                for node_id, row in nodes.iterrows()
            }
            curveness = 0
            break
    else:
        pos = nx.spring_layout(
            graph.to_nx(),
            k=1.2,
            iterations=500,
            seed=42,
        )

        curveness = 0.15
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
                "label": {},
                "data": [
                    {
                        "id": str(n.id),
                        "x": float(pos[n.id][0]),
                        "y": float(pos[n.id][1]),
                        # Adjust node size
                        "symbolSize": 22 if n.role == "target" else 6,
                        "symbol": {
                            "source": "circle",
                            "target": "circle",
                            "both": "diamond",
                            "isolated": "square",
                        }.get(n.role, "circle"),
                        "itemStyle": {
                            "color": n.color,
                            "borderColor": "#666666",
                            "borderWidth": 1,
                            "opacity": 0.9,
                        },
                        "label": {
                        "show": n.role == "target",
                        "position": "inside",
                        "fontSize": 8,
                        "color": "#000000",
                    },

                        "name": str(n.id),
                        "value": str(n.role),
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


