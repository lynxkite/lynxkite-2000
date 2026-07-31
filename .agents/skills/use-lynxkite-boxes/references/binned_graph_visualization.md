**Binned graph visualization:**
Nodes binned together by x and y are aggregated into one node.
Edges between bins are aggregated into one edge.
```python
@op("Binned graph visualization", view="matplotlib", color="blue", icon="table")
def binned_graph_visualization(
    self,
    b: core.Bundle,
    *,
    x_property: str,
    y_property: str,
    x_bins=5,
    y_bins=5,
    show_loops: bool = False,
):
    """
    Nodes binned together by x and y are aggregated into one node.
    Edges between bins are aggregated into one edge.
    """
    b = b.copy()
    (nodes, node_id), (edges, source_id, target_id) = _nodes_and_edges(b)

    nodes["x_bin"] = pd.cut(nodes[x_property], bins=x_bins)
    nodes["y_bin"] = pd.cut(nodes[y_property], bins=y_bins)

    # Compute node counts per bin.
    bin_counts = nodes.groupby(["x_bin", "y_bin"], observed=True).size().reset_index(name="count")
    bin_counts["key"] = bin_counts.apply(lambda row: f"{row['x_bin']},{row['y_bin']}", axis=1)
    # Assign each node to its bin.
    nodes["bin"] = list(zip(nodes["x_bin"], nodes["y_bin"]))
    nodes["bin_key"] = nodes["bin"].apply(lambda b: f"{b[0]},{b[1]}")
    # Aggregate edges between bins.
    node_bins = nodes.set_index(node_id)["bin_key"]
    edges["source_bin"] = edges[source_id].map(node_bins)
    edges["target_bin"] = edges[target_id].map(node_bins)
    edges = edges.dropna(subset=["source_bin", "target_bin"])

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

```
