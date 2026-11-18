"""Custom operations for the GFM."""

import numpy as np
import pandas as pd
from lynxkite_core.ops import op_registration
from lynxkite_graph_analytics import core

op = op_registration(core.ENV, "GFM")


CONV_NAMES = ["GCNConv", "SAGEConv", "GraphConv"]
ACT_NAMES = ["tanh", "sigmoid", "relu", "sin"]


@op("Generate world", color="green")
def generate_world(
    *, n: int, p: float, seed: int, d: int, max_layers: int = 3, max_noise: float = 0.1
) -> core.Bundle:
    """
    Generate a random graph with n nodes and edge probability p,
    with every node having a d-dimensional random vector.
    """
    rng = np.random.default_rng(seed)

    src, dst = np.triu_indices(n, k=1)
    keep = rng.random(len(src)) < p
    edges = pd.DataFrame({"source": src[keep], "target": dst[keep]})

    nodes = pd.DataFrame({"id": np.arange(n), "x": list(rng.normal(size=(n, d)))})

    n_layers = int(rng.integers(1, max_layers + 1))
    process = dict(
        convs=[str(c) for c in rng.choice(CONV_NAMES, n_layers)],
        acts=[str(a) for a in rng.choice(ACT_NAMES, n_layers)],
        alpha=float(rng.uniform(0.1, 0.6)),
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
