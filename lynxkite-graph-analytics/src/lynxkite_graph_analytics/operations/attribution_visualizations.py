"""Attribution visualizations exposed as boxes."""

from lynxkite_core import ops

from .. import core
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

op = ops.op_registration(core.ENV)

# Global seed for sampled subgraph layout/choices
SAMPLE_SEED = 42


def _build_sampled_graph(edge_index, sampled_nodes):
    """Build a NetworkX graph for the sampled node list and return mapping.

    Returns (G, pos, node_mapping, edge_to_orig_idx)
    """
    if hasattr(edge_index, "cpu"):
        edge_index_np = edge_index.cpu().numpy()
    else:
        edge_index_np = np.asarray(edge_index)
    edge_index_np = np.array(edge_index_np, dtype=int)

    node_mapping = {old: new for new, old in enumerate(sampled_nodes)}
    edge_list = []
    edge_to_orig_idx = {}
    for orig_edge_idx, (i, j) in enumerate(edge_index_np.T):
        if i in sampled_nodes and j in sampled_nodes:
            new_edge = (node_mapping[i], node_mapping[j])
            edge_list.append(list(new_edge))
            edge_to_orig_idx[new_edge] = orig_edge_idx
            edge_to_orig_idx[(node_mapping[j], node_mapping[i])] = orig_edge_idx

    G = nx.Graph()
    G.add_nodes_from(range(len(sampled_nodes)))
    if edge_list:
        G.add_edges_from(edge_list)
    pos = nx.spring_layout(G, seed=SAMPLE_SEED, k=1.0)
    return G, pos, node_mapping, edge_to_orig_idx


def node_coloring_plot(edge_index, attr_tensor, sampled_nodes, figsize=(6, 6)):
    G, pos, _, _ = _build_sampled_graph(edge_index, sampled_nodes)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    if len(sampled_nodes) == 0:
        ax.axis("off")
        return fig, [ax]

    attr_np_full = attr_tensor.detach().cpu().numpy()
    attr_np = np.array(
        [attr_np_full[node] if node < len(attr_np_full) else 0 for node in sampled_nodes]
    )
    amin, amax = attr_np.min(), attr_np.max()
    if amax > amin:
        attr_norm = (attr_np - amin) / (amax - amin)
    else:
        attr_norm = np.ones_like(attr_np) * 0.5

    nx.draw_networkx_edges(G, pos, width=2.0, alpha=0.5, edge_color="gray", ax=ax)
    cmap = plt.get_cmap("YlOrRd")
    node_colors = [cmap(attr_norm[i]) for i in range(len(sampled_nodes))]
    node_sizes = [200 + attr_norm[i] * 800 for i in range(len(sampled_nodes))]
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.8,
        edgecolors="black",
        linewidths=1,
        ax=ax,
    )
    labels = {i: str(sampled_nodes[i]) for i in range(len(sampled_nodes))}
    nx.draw_networkx_labels(G, pos, labels, font_size=7, ax=ax)
    ax.axis("off")
    plt.tight_layout()
    return fig, [ax]


def edge_attention_plot(edge_index, attention_tensor, sampled_nodes, figsize=(6, 6)):
    G, pos, node_mapping, edge_to_orig_idx = _build_sampled_graph(edge_index, sampled_nodes)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    if len(G.edges()) == 0:
        ax.axis("off")
        return fig, [ax]

    att_np = (
        attention_tensor.detach().cpu().numpy()
        if hasattr(attention_tensor, "detach")
        else np.array(attention_tensor)
    )
    if len(att_np.shape) > 1:
        att_np = att_np.mean(axis=1)

    edge_attention_values = []
    for u, v in G.edges():
        if (u, v) in edge_to_orig_idx:
            orig_idx = edge_to_orig_idx[(u, v)]
            if orig_idx < len(att_np):
                edge_attention_values.append(att_np[orig_idx])
            else:
                edge_attention_values.append(0.0)
        else:
            edge_attention_values.append(0.0)
    edge_attention_values = np.array(edge_attention_values)

    if edge_attention_values.size == 0:
        ax.axis("off")
        return fig, [ax]

    edge_attn_norm = (edge_attention_values - edge_attention_values.min()) / (
        edge_attention_values.max() - edge_attention_values.min() + 1e-8
    )
    edges = list(G.edges())
    for i, (u, v) in enumerate(edges):
        attn_val = edge_attn_norm[i]
        width = 2.0 + attn_val * 5.0
        color = plt.get_cmap("YlOrRd")(0.3 + attn_val * 0.7)
        ax.plot(
            [pos[u][0], pos[v][0]],
            [pos[u][1], pos[v][1]],
            color=color,
            linewidth=width,
            alpha=0.8,
            zorder=1,
        )

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color="lightgray",
        node_size=300,
        alpha=0.6,
        edgecolors="black",
        linewidths=1,
        ax=ax,
    )
    labels = {i: str(sampled_nodes[i]) for i in range(len(sampled_nodes))}
    nx.draw_networkx_labels(G, pos, labels, font_size=7, ax=ax)
    ax.axis("off")
    plt.tight_layout()
    return fig, [ax]


def _sample_nodes_for_plot(edge_index, first_attr, num_cells: int = 25):
    """Return a list of sampled node indices (connected component around high-attr node)."""

    edge_index_np = edge_index.cpu().numpy()
    total_nodes = max(edge_index_np.max() + 1, len(first_attr))

    full_graph = nx.Graph()
    full_graph.add_nodes_from(range(total_nodes))
    full_graph.add_edges_from(edge_index_np.T)

    sorted_by_attr = np.argsort(first_attr)[::-1]
    start_node = None
    for node in sorted_by_attr:
        if full_graph.degree(node) > 0:
            start_node = int(node)
            break
    if start_node is None:
        for node in range(total_nodes):
            if full_graph.degree(node) > 0:
                start_node = node
                break
        if start_node is None:
            start_node = 0

    sampled_nodes = set([start_node])
    frontier = set([start_node])
    while len(sampled_nodes) < num_cells and frontier:
        new_frontier = set()
        for node in frontier:
            neighbors = set(full_graph.neighbors(node))
            candidates = neighbors - sampled_nodes
            if candidates:
                sorted_candidates = sorted(
                    candidates,
                    key=lambda n: first_attr[n] if n < len(first_attr) else 0,
                    reverse=True,
                )
                to_add = sorted_candidates[: max(1, (num_cells - len(sampled_nodes)) // 2)]
                new_frontier.update(to_add)
                sampled_nodes.update(to_add)
        frontier = new_frontier

    # ensure items are ints for typed sorting; filter out any None
    return sorted([int(n) for n in sampled_nodes if n is not None])[:num_cells]


def _get_common_first_attr(edge_index, feat_norm, pred_conf, degree):
    """Return a common first_attr array used to sample the same subgraph for all views.

    Preference order: `feat_norm`, `degree`, `pred_conf`. Falls back to ones.
    """
    if feat_norm is not None:
        return feat_norm.detach().cpu().numpy()
    if degree is not None:
        return degree.detach().cpu().numpy()
    if pred_conf is not None:
        return pred_conf.detach().cpu().numpy()
    try:
        total_nodes = int(edge_index.max().item() + 1)
    except Exception:
        total_nodes = 0
    return np.ones(total_nodes)


def _get_attrs_from_bundle(graph):
    """Try to read precomputed attributes from the Bundle. Returns
    (edge_index, feat_norm, pred_conf, degree, attention_tensor) where any
    value can be None if not present.
    """
    import torch

    nodes = graph.dfs.get("nodes")
    edges = graph.dfs.get("edges")
    if nodes is None or edges is None:
        return None, None, None, None, None

    edge_index = torch.tensor(edges[["source", "target"]].to_numpy().T, dtype=torch.long)

    feat_norm = None
    pred_conf = None
    degree = None
    attention_tensor = None

    if "feat_norm" in nodes.columns:
        feat_norm = torch.tensor(nodes["feat_norm"].to_numpy())
    if "pred_conf" in nodes.columns:
        pred_conf = torch.tensor(nodes["pred_conf"].to_numpy())
    if "degree" in nodes.columns:
        degree = torch.tensor(nodes["degree"].to_numpy())
    if "attention" in edges.columns:
        attention_tensor = torch.tensor(edges["attention"].to_numpy())

    return edge_index, feat_norm, pred_conf, degree, attention_tensor


@op("Attribution - Edge attention", view="matplotlib", icon="eye", color="purple")
def attribution_edge_attention(graph: core.Bundle, *, num_cells: int = 25):
    """Box showing edge attention via edge thickness/color."""
    edge_index, feat_norm, pred_conf, degree, attention_tensor = _get_attrs_from_bundle(graph)
    if edge_index is None or attention_tensor is None:
        raise RuntimeError("Run the GAT compute box first to populate attention on edges.")

    common_first_attr = _get_common_first_attr(edge_index, feat_norm, pred_conf, degree)
    sampled_nodes = _sample_nodes_for_plot(edge_index, common_first_attr, num_cells=num_cells)
    fig, axes = edge_attention_plot(edge_index, attention_tensor, sampled_nodes, figsize=(6, 6))
    return fig


@op("Attribution - Degree", view="matplotlib", icon="eye", color="purple")
def attribution_degree(graph: core.Bundle, *, num_cells: int = 25):
    """Visualize node degree as node coloring/size."""
    edge_index, feat_norm, pred_conf, degree, attention = _get_attrs_from_bundle(graph)
    if edge_index is None or degree is None:
        raise RuntimeError("Run the GAT compute box first to populate node degrees.")
    common_first_attr = _get_common_first_attr(edge_index, feat_norm, pred_conf, degree)
    sampled_nodes = _sample_nodes_for_plot(edge_index, common_first_attr, num_cells=num_cells)
    fig, axes = node_coloring_plot(edge_index, degree, sampled_nodes, figsize=(6, 6))
    return fig


@op("Attribution - Feature magnitude", view="matplotlib", icon="eye", color="purple")
def attribution_feature_magnitude(graph: core.Bundle, *, num_cells: int = 25):
    edge_index, feat_norm, pred_conf, degree, attention = _get_attrs_from_bundle(graph)
    if edge_index is None or feat_norm is None:
        raise RuntimeError("Run the GAT compute box first to populate feature magnitudes.")
    common_first_attr = _get_common_first_attr(edge_index, feat_norm, pred_conf, degree)
    sampled_nodes = _sample_nodes_for_plot(edge_index, common_first_attr, num_cells=num_cells)
    fig, axes = node_coloring_plot(edge_index, feat_norm, sampled_nodes, figsize=(6, 6))
    return fig


@op("Attribution - Prediction confidence", view="matplotlib", icon="eye", color="purple")
def attribution_prediction_confidence(graph: core.Bundle, *, num_cells: int = 25):
    edge_index, feat_norm, pred_conf, degree, attention = _get_attrs_from_bundle(graph)
    if edge_index is None or pred_conf is None:
        raise RuntimeError("Run the GAT compute box first to populate prediction confidence.")
    common_first_attr = _get_common_first_attr(edge_index, feat_norm, pred_conf, degree)
    sampled_nodes = _sample_nodes_for_plot(edge_index, common_first_attr, num_cells=num_cells)
    fig, axes = node_coloring_plot(edge_index, pred_conf, sampled_nodes, figsize=(6, 6))
    return fig


@op("Attribution - Node attention", view="matplotlib", icon="eye", color="purple")
def attribution_node_attention(graph: core.Bundle, *, num_cells: int = 25):
    """Visualize per-node attention (sum of incident edge attentions)."""
    edge_index, feat_norm, pred_conf, degree, attention = _get_attrs_from_bundle(graph)
    if edge_index is None or attention is None:
        raise RuntimeError("Run the GAT compute box first to populate attention on edges.")

    import torch

    # edge attention per edge -> aggregate to nodes by summing incident edges
    att_np = (
        attention.detach().cpu().numpy() if hasattr(attention, "detach") else np.array(attention)
    )
    if len(att_np.shape) > 1:
        att_np = att_np.mean(axis=1)

    edge_index_np = edge_index.cpu().numpy()
    total_nodes = int(edge_index.max().item() + 1)
    node_att = np.zeros(total_nodes)
    for idx, (u, v) in enumerate(edge_index_np.T):
        val = float(att_np[idx]) if idx < len(att_np) else 0.0
        node_att[int(u)] += val
        node_att[int(v)] += val

    node_att_tensor = torch.tensor(node_att)

    common_first_attr = _get_common_first_attr(edge_index, feat_norm, pred_conf, degree)
    sampled_nodes = _sample_nodes_for_plot(edge_index, common_first_attr, num_cells=num_cells)
    fig, axes = node_coloring_plot(edge_index, node_att_tensor, sampled_nodes, figsize=(6, 6))
    return fig
