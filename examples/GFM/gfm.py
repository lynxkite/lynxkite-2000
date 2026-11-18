"""Custom operations for the GFM."""

import torch

import numpy as np
import pandas as pd
from torch import nn
from torch_geometric.nn import GraphConv
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import GCNConv
from lynxkite_core.ops import op_registration
from lynxkite_graph_analytics import core

op = op_registration(core.ENV, "GFM")


CONV_NAMES = ["GCNConv", "SAGEConv", "GraphConv"]
ACT_NAMES = ["tanh", "sigmoid", "relu", "sin"]


def stack_vectors(col: pd.Series) -> np.ndarray:
    return np.stack([np.asarray(v, dtype=float) for v in col])


@op("Generate world", color="green")
def generate_world(
    *,
    n: int,
    p: float,
    seed: int,
    d: int,
    max_layers: int = 3,
    max_noise: float = 0.1,
    node_type_count: int = 3,
    edge_type_count: int = 3,
) -> core.Bundle:
    """
    Generate a random graph with n nodes and edge probability p,
    with every node having a d-dimensional random vector.
    """
    rng = np.random.default_rng(seed)

    src, dst = np.triu_indices(n, k=1)
    keep = rng.random(len(src)) < p

    n_nt = int(rng.integers(1, node_type_count + 1))
    n_et = int(rng.integers(1, edge_type_count + 1))

    edges = pd.DataFrame(
        {"source": src[keep], "target": dst[keep], "type": rng.integers(n_et, size=keep.sum())}
    )
    nodes = pd.DataFrame(
        {"id": np.arange(n), "type": rng.integers(n_nt, size=n), "x": list(rng.normal(size=(n, d)))}
    )

    n_layers = int(rng.integers(1, max_layers + 1))
    process = dict(
        n_node_types=n_nt,
        n_edge_types=n_et,
        convs=[str(c) for c in rng.choice(CONV_NAMES, n_layers)],
        acts=[str(a) for a in rng.choice(ACT_NAMES, n_layers)],
        alpha=[float(a) for a in rng.uniform(0.1, 0.6, n_nt)],
        noise=float(rng.uniform(0.0, max_noise)),
        weight_seed=int(rng.integers(2**31)),
    )

    b = core.Bundle()
    b.dfs["nodes"], b.dfs["edges"] = nodes, edges
    b.relations = [
        core.RelationDefinition(
            name="Random graph",
            df="edges",
            source_column="source",
            target_column="target",
            source_table="nodes",
            target_table="nodes",
            source_key="id",
            target_key="id",
        )
    ]
    b.other["process"] = process
    return b


CONVS = {"GCNConv": GCNConv, "SAGEConv": SAGEConv, "GraphConv": GraphConv}
ACTS = {"tanh": torch.tanh, "sigmoid": torch.sigmoid, "relu": torch.relu, "sin": torch.sin}


class TypeLayer(nn.Module):
    def __init__(self, conv_name, d, n_et, n_nt):
        super().__init__()
        self.convs = nn.ModuleList(
            CONVS[conv_name](in_channels=d, out_channels=d) for _ in range(n_et)
        )
        self.self_lin = nn.ModuleList(nn.Linear(d, d) for _ in range(n_nt))

    def forward(self, h, edge_indices, node_type):
        out = sum(conv(h, ei) for conv, ei in zip(self.convs, edge_indices))
        self_out = torch.stack([lin(h) for lin in self.self_lin])
        return out + self_out[node_type, torch.arange(len(h))]


@op("Simulate", color="orange")
def simulate(b: core.Bundle, *, steps: int, seed: int) -> core.Bundle:
    spec = b.other["process"]
    nodes, edges = b.dfs["nodes"], b.dfs["edges"]
    x = torch.tensor(stack_vectors(nodes["x"]), dtype=torch.float32)
    ntype = torch.tensor(nodes["type"].to_numpy(), dtype=torch.long)

    eis = []
    for t in range(spec["n_edge_types"]):
        e = edges[edges["type"] == t]
        ei = torch.tensor(e[["source", "target"]].to_numpy().T, dtype=torch.long)
        eis.append(torch.cat([ei, ei.flip(0)], dim=1))

    torch.manual_seed(spec["weight_seed"])
    d = x.shape[1]
    layers = nn.ModuleList(
        TypeLayer(c, d, spec["n_edge_types"], spec["n_node_types"]) for c in spec["convs"]
    )
    acts = [ACTS[a] for a in spec["acts"]]
    alpha = torch.tensor(spec["alpha"])[ntype].unsqueeze(1)
    g = torch.Generator().manual_seed(seed)

    history = [x]
    with torch.no_grad():
        for _ in range(steps):
            h = x
            for layer, act in zip(layers, acts):
                h = act(layer(h, eis, ntype))
            x = (1 - alpha) * x + alpha * torch.tanh(h)
            x = x + spec["noise"] * torch.randn(x.shape, generator=g)
            history.append(x)

    n = len(nodes)
    states = pd.DataFrame(
        {
            "step": np.repeat(np.arange(steps + 1), n),
            "id": np.tile(nodes["id"].to_numpy(), steps + 1),
            "x": [row for h in history for row in h.numpy()],
        }
    )
    out = core.Bundle(dfs=dict(b.dfs), relations=list(b.relations), other=dict(b.other))
    out.dfs["states"] = states
    return out


@op("Observe", color="blue")
def observe(
    b: core.Bundle, *, k_steps: int = 10, p_apply: float = 0.5, seed: int = 0
) -> core.Bundle:
    rng = np.random.default_rng(seed)
    spec = b.other["process"]
    nodes, edges, states = b.dfs["nodes"], b.dfs["edges"], b.dfs["states"]
    ids = nodes["id"].to_numpy()
    log = {}

    def on(name):
        log[name] = bool(rng.random() < p_apply)
        return log[name]

    kept = np.ones(len(ids), bool)
    obs_edges = edges.copy()

    if on("missing_nodes_edges"):
        fn, fe = rng.uniform(0, 0.4, 2)
        kept &= rng.random(len(ids)) >= fn
        obs_edges = obs_edges[rng.random(len(obs_edges)) >= fe]
        log["frac_nodes_dropped"], log["frac_edges_dropped"] = float(fn), float(fe)

    if on("missing_types"):
        if rng.random() < 0.5:
            t = int(rng.integers(spec["n_edge_types"]))
            obs_edges = obs_edges[obs_edges["type"] != t]
            log["dropped_edge_type"] = t
        else:
            t = int(rng.integers(spec["n_node_types"]))
            kept &= nodes["type"].to_numpy() != t
            log["dropped_node_type"] = t

    kept_ids = ids[kept]
    obs_edges = obs_edges[obs_edges["source"].isin(kept_ids) & obs_edges["target"].isin(kept_ids)]

    if on("graph_edits") and len(kept_ids) > 1:
        r, a = rng.uniform(0, 0.3, 2)
        obs_edges = obs_edges.sample(frac=1 - r, random_state=seed)
        n_add = int(a * len(edges))
        u, v = rng.choice(kept_ids, n_add), rng.choice(kept_ids, n_add)
        new = pd.DataFrame(
            {
                "source": np.minimum(u, v),
                "target": np.maximum(u, v),
                "type": rng.integers(spec["n_edge_types"], size=n_add),
            }
        )
        obs_edges = pd.concat([obs_edges, new[new["source"] != new["target"]]])
        obs_edges = obs_edges.drop_duplicates(["source", "target"])
        log["frac_edges_removed"], log["frac_edges_added"] = float(r), float(a)

    obs_edges = obs_edges.reset_index(drop=True)
    obs_nodes = nodes[nodes["id"].isin(kept_ids)][["id", "type"]].reset_index(drop=True)

    s = states[(states["step"] < k_steps) & states["id"].isin(kept_ids)].reset_index(drop=True)
    x = stack_vectors(s["x"]).copy()
    mask = np.ones_like(x)

    if on("missing_variable"):
        j = int(rng.integers(x.shape[1]))
        x[:, j], mask[:, j] = 0, 0
        log["hidden_dim"] = j

    if on("noise"):
        sigma = float(np.exp(rng.uniform(np.log(0.01), np.log(0.3))) * x.std())
        x = x + sigma * rng.normal(size=x.shape) * mask
        log["noise_sigma"] = sigma

    obs_states = pd.DataFrame({"step": s["step"], "id": s["id"], "x": list(x), "mask": list(mask)})

    out = core.Bundle(dfs=dict(b.dfs), relations=list(b.relations), other=dict(b.other))
    out.dfs.update(obs_nodes=obs_nodes, obs_edges=obs_edges, obs_states=obs_states)
    out.other["observation"] = log
    return out


@op("Final graph")
def final_graph(b: core.Bundle, *, step: int = -1) -> core.Bundle:
    s = b.dfs["states"]
    t = s["step"].max() if step < 0 else step
    cur = s[s["step"] == t].sort_values("id")

    nodes = b.dfs["nodes"].copy()
    nodes["x"] = list(cur["x"].to_numpy())

    out = core.Bundle(dfs={"nodes": nodes, "edges": b.dfs["edges"]})
    out.relations = list(b.relations)
    return out
