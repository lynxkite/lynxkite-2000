"""Custom operations for the GFM."""

import numpy as np
import pandas as pd
from lynxkite_core.ops import op_registration
from lynxkite_graph_analytics import core

op = op_registration(core.ENV, "GFM")


@op("Generate world", color="green")
def generate_world(*, n: int, p: float, seed: int, d: int) -> core.Bundle:
    """
    Generate a random graph with n nodes and edge probability p,
    with every node having a d-dimensional random vector.
    """
    rng = np.random.default_rng(seed)

    src, dst = np.triu_indices(n, k=1)
    keep = rng.random(len(src)) < p
    edges = pd.DataFrame({"source": src[keep], "target": dst[keep]})

    x = rng.normal(size=(n, d))
    nodes = pd.DataFrame({"id": np.arange(n), "x": list(x)})

    b = core.Bundle()
    b.dfs["nodes"] = nodes
    b.dfs["edges"] = edges
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
    return b
