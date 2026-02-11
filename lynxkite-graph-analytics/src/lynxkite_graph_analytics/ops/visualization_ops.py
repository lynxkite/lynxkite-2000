"""Visualizations."""

from lynxkite_core import ops

from .. import core
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


@op("Scatter plot", icon="chart-dots", color="blue", view="matplotlib")
def scatter_plot(b: core.Bundle, *, x: core.TableColumn, y: core.TableColumn):
    table_x, column_x = x
    table_y, column_y = y
    dx = b.dfs[table_x][column_x]
    dy = b.dfs[table_y][column_y]
    correlation = dx.corr(dy)
    plt.figure(figsize=(6, 6))
    sns.regplot(x=dx, y=dy)
    plt.title(f"Correlation: {correlation:.2f}")
    plt.xlabel(column_x)
    plt.ylabel(column_y)


@op("Binned graph visualization", view="matplotlib", color="blue", icon="table")
def binned_graph_visualization(
    b: core.Bundle,
    *,
    x_property: core.NodePropertyName,
    y_property: core.NodePropertyName,
    x_bins=5,
    y_bins=5,
    show_loops: bool = False,
):
    """
    Nodes binned together by x and y are aggregated into one node.
    Edges between bins are aggregated into one edge.
    """
    nodes = b.dfs["nodes"].copy()
    edges = b.dfs["edges"].copy()
    if "weight" not in edges.columns:
        edges["weight"] = 1

    nodes["x_bin"] = pd.cut(nodes[x_property], bins=x_bins)
    nodes["y_bin"] = pd.cut(nodes[y_property], bins=y_bins)

    # Compute node counts per bin.
    bin_counts = nodes.groupby(["x_bin", "y_bin"], observed=True).size().reset_index(name="count")
    bin_counts["key"] = bin_counts.apply(lambda row: f"{row['x_bin']},{row['y_bin']}", axis=1)
    # Assign each node to its bin.
    nodes["bin"] = list(zip(nodes["x_bin"], nodes["y_bin"]))
    nodes["bin_key"] = nodes["bin"].apply(lambda b: f"{b[0]},{b[1]}")
    # Aggregate edges between bins.
    edges["source_bin"] = nodes.loc[edges["source"], "bin_key"].values
    edges["target_bin"] = nodes.loc[edges["target"], "bin_key"].values

    edge_counts = (
        edges.groupby(["source_bin", "target_bin"], observed=True).size().reset_index(name="weight")
    )

    # Build network.
    G = nx.DiGraph()
    for _, row in bin_counts.iterrows():
        G.add_node(row["key"], count=row["count"])

    for _, row in edge_counts.iterrows():
        if show_loops or row["source_bin"] != row["target_bin"]:
            G.add_edge(row["source_bin"], row["target_bin"], weight=row["weight"])

    # Compute node positions.
    def bin_center(interval):
        return (interval.left + interval.right) / 2

    pos = {
        row["key"]: (bin_center(row["x_bin"]), bin_center(row["y_bin"]))
        for _, row in bin_counts.iterrows()
    }
    # Node sizes.
    size = bin_counts["count"] ** 0.5  # Circle area proportional to count.
    max_size = size.max()
    node_sizes = [s * 1000 / max_size for s in size]

    # Edge widths.
    max_weight = edge_counts["weight"].max()
    edge_widths = [10 * (w / max_weight) for w in [G[u][v]["weight"] for u, v in G.edges()]]

    # Start plotting.
    plt.figure(figsize=(8, 8))

    # Edges.
    nx.draw_networkx_edges(
        G,
        pos,
        connectionstyle="arc3,rad=0.3",
        width=edge_widths,
        edge_color="#dddddd",
        node_size=[s + 1000 for s in node_sizes],
    )

    # Nodes.
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="#00c0ff")

    # Labels.
    for node, (x_pos, y_pos) in pos.items():
        if G.nodes[node]["count"]:
            plt.text(
                x_pos,
                y_pos,
                str(G.nodes[node]["count"]),
                fontsize=8,
                ha="center",
                va="center",
                color="black",
                fontweight="bold",
            )

    # Axis labels and grid.
    x_intervals = nodes["x_bin"].cat.categories
    y_intervals = nodes["y_bin"].cat.categories
    x_edges = np.unique([i.left for i in x_intervals] + [x_intervals[-1].right])
    y_edges = np.unique([i.left for i in y_intervals] + [y_intervals[-1].right])
    ax = plt.gca()
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.xticks(x_edges, [f"{v:.2f}" for v in x_edges])
    plt.yticks(y_edges, [f"{v:.2f}" for v in y_edges])
    plt.xlabel(x_property)
    plt.ylabel(y_property)
    plt.grid(True, color="#00c0ff", alpha=0.3)
    plt.xlim(x_edges[0], x_edges[-1])
    plt.ylim(y_edges[0], y_edges[-1])
