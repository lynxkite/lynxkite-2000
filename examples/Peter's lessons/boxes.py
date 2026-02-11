import enum
import pandas as pd
import networkx as nx
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from lynxkite_core.ops import op, LongStr
import lynxkite_graph_analytics as lk


@op(lk.ENV, "Graph from edge list", color="green", icon="topology-star-3")
def graph_from_edge_list(
    df: pd.DataFrame, *, source: lk.DataFrameColumn, target: lk.DataFrameColumn
) -> lk.Bundle:
    b = lk.Bundle()
    b.dfs["nodes"] = pd.DataFrame({"id": pd.concat([df[source], df[target]]).unique()})
    b.dfs["edges"] = df.rename(columns={source: "source", target: "target"})
    b.relations.append(
        lk.RelationDefinition(
            name="graph",
            df="edges",
            source_column="source",
            source_table="nodes",
            source_key="id",
            target_column="target",
            target_table="nodes",
            target_key="id",
        )
    )
    return b


@op(lk.ENV, "NetworkX", "Degree", icon="topology-star-3")
def degree(g: nx.Graph) -> nx.Graph:
    g = g.copy()
    nx.set_node_attributes(g, name="degree", values=dict(g.degree()))
    return g


class AggregationMethod(enum.StrEnum):
    sum = "sum"
    mean = "mean"
    max = "max"
    min = "min"

    def apply(self, values):
        if self == AggregationMethod.sum:
            return np.sum(values)
        elif self == AggregationMethod.mean:
            return np.mean(values)
        elif self == AggregationMethod.max:
            return np.max(values)
        elif self == AggregationMethod.min:
            return np.min(values)
        else:
            raise ValueError(f"Unsupported aggregation method: {self}")


@op(lk.ENV, "Aggregate on neighbors", icon="topology-star-3")
def aggregate_on_neighbors(
    g: nx.Graph, *, property: lk.NodePropertyName, aggregation: AggregationMethod
) -> nx.Graph:
    g = g.copy()
    for node in g.nodes:
        neighbor_values = [g.nodes[neighbor].get(property, 0) for neighbor in g.neighbors(node)]
        if not neighbor_values:
            continue
        agg_value = aggregation.apply(neighbor_values)
        g.nodes[node][f"{property}_neighborhood_{aggregation}"] = agg_value
    return g


@op(lk.ENV, "Derive property", icon="arrow-big-right-lines")
def derive_property(b: lk.Bundle, *, table_name: lk.TableName, formula: LongStr) -> lk.Bundle:
    b = b.copy()
    df = b.dfs[table_name]
    b.dfs[table_name] = df.eval(formula)
    return b


@op(lk.ENV, "Scatter plot", icon="chart-dots", color="blue", view="matplotlib")
def scatter_plot(b: lk.Bundle, *, x: lk.TableColumn, y: lk.TableColumn):
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


@op(lk.ENV, "Binned graph visualization", view="matplotlib", color="blue", icon="table")
def binned_graph_visualization(
    b: lk.Bundle,
    *,
    x_property: lk.NodePropertyName,
    y_property: lk.NodePropertyName,
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
