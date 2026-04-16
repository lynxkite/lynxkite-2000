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


@op("Bar chart", icon="chart-bar", color="blue", view="matplotlib")
def bar_chart(
    b: core.Bundle,
    *,
    x: core.TableColumn,
    y: core.TableColumn,
):
    table_x, column_x = x
    table_y, column_y = y
    if table_x == table_y:
        df = b.dfs[table_x]
    else:
        df = b.dfs[table_x].merge(b.dfs[table_y], left_index=True, right_index=True)
    sorted_df = df.sort_values(column_x)
    dx = sorted_df[column_x]
    dy = sorted_df[column_y]
    plt.figure(figsize=(6, 6))
    sns.barplot(x=dx, y=dy)
    plt.xlabel(column_x)
    plt.ylabel(column_y)


@op("Histogram", icon="chart-histogram", color="blue", view="matplotlib")
def histogram(b: core.Bundle, *, column: core.TableColumn, bins: int = 20):
    table, col = column
    data = b.dfs[table][col]
    plt.figure(figsize=(6, 6))
    sns.histplot(data, bins=bins)
    plt.xlabel(col)
    plt.ylabel("Count")


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
