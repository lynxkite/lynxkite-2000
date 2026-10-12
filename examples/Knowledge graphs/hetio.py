"""Custom operations for the Hetionet knowledge graph."""

import pandas as pd
from lynxkite_core.ops import op_registration
from lynxkite_graph_analytics import core

op = op_registration(core.ENV, "Knowledge graphs")


@op("Import hetionet", slow=True, color="green", icon="graph")
def import_hetionet():
    """Import Hetionet data from online."""
    b = core.Bundle()
    nodes_url = (
        "https://raw.githubusercontent.com/hetio/hetionet/main/hetnet/tsv/hetionet-v1.0-nodes.tsv"
    )
    edges_url = "https://media.githubusercontent.com/media/hetio/hetionet/main/hetnet/tsv/hetionet-v1.0-edges.sif.gz"
    nodes_df = pd.read_csv(nodes_url, sep="\t")
    edges_df = pd.read_csv(edges_url, sep="\t", compression="gzip")

    b.dfs["nodes"] = nodes_df
    b.dfs["edges"] = edges_df
    b.relations.append(
        core.RelationDefinition(
            df="edges",
            source_column="source",
            target_column="target",
            source_table="nodes",
            target_table="nodes",
            source_key="id",
            target_key="id",
            name="Hetionet",
        )
    )
    return b
