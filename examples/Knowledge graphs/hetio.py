"""Custom operations for the Hetionet knowledge graph."""

import typing
from enum import StrEnum

import pandas as pd
from lynxkite_core.ops import op_registration
from lynxkite_graph_analytics import core

op = op_registration(core.ENV, "Knowledge graphs")


class Kind(StrEnum):
    Anatomy = "Anatomy"
    BiologicalProcess = "BiologicalProcess"
    CellularComponent = "CellularComponent"
    Compound = "Compound"
    Disease = "Disease"
    Gene = "Gene"
    MolecularFunction = "MolecularFunction"
    Pathway = "Pathway"
    PharmacologicClass = "PharmacologicClass"
    SideEffect = "SideEffect"
    Symptom = "Symptom"


class Metaedge(StrEnum):
    AdG = "AdG"
    AeG = "AeG"
    AuG = "AuG"
    CbG = "CbG"
    CcSE = "CcSE"
    CdG = "CdG"
    CpD = "CpD"
    CrC = "CrC"
    CtD = "CtD"
    CuG = "CuG"
    DaG = "DaG"
    DdG = "DdG"
    DlA = "DlA"
    DpS = "DpS"
    DrD = "DrD"
    DuG = "DuG"
    GcG = "GcG"
    GiG = "GiG"
    GpBP = "GpBP"
    GpCC = "GpCC"
    GpMF = "GpMF"
    GpPW = "GpPW"
    GrG = "Gr>G"
    PCiC = "PCiC"


MetaedgeDropdown = typing.Annotated[
    list[str],
    {
        "format": "multi-dropdown_from_list",
        "options": [metaedge.value for metaedge in Metaedge],
    },
]

KindDropdown = typing.Annotated[
    list[str],
    {
        "format": "multi-dropdown_from_list",
        "options": [kind.value for kind in Kind],
    },
]


@op("Hetionet lookup")
def hetionet_lookup(
    b: core.Bundle, *, name: str, neighbour_kinds: KindDropdown, metaedges: MetaedgeDropdown
):
    """Keep only nodes with the given name and their neighbors of the given kinds, connected by the given metaedges."""
    b = b.copy()
    all_nodes, all_edges = b.dfs["nodes"], b.dfs["edges"]

    seeds = all_nodes[all_nodes["name"].str.lower() == name.lower()]
    seed_ids = set(seeds["id"])

    edges = all_edges
    if metaedges:
        edges = edges[edges["metaedge"].isin(metaedges)]
    edges = edges[edges["source"].isin(seed_ids) | edges["target"].isin(seed_ids)]

    neighbor_ids = (set(edges["source"]) | set(edges["target"])) - seed_ids
    neighbors = all_nodes[all_nodes["id"].isin(neighbor_ids)]
    if neighbour_kinds:
        neighbors = neighbors[neighbors["kind"].isin(neighbour_kinds)]

    keep = seed_ids | set(neighbors["id"])
    b.dfs["nodes"] = all_nodes[all_nodes["id"].isin(keep)]
    b.dfs["edges"] = edges[edges["source"].isin(keep) & edges["target"].isin(keep)]
    return b


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
