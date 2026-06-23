"""Operations for tables."""

import enum

import networkx as nx

from lynxkite_core import ops
from .. import core

op = ops.op_registration(core.ENV, "Segmentation operations")


class EdgeDirection(enum.StrEnum):
    Ignore = "Ignore directions"
    Both = "Require both directions"


@op("Find Connected Components", icon="filter-filled")
def sample_table(b: core.Bundle, *, edge_direction: EdgeDirection, segmentation_name: str):
    b = b.copy()

    graph, meta = b.to_nx_meta()

    colum_names = set()
    for r in b.relations:
        colum_names.update(b.dfs[r.source_table].columns.values)
        colum_names.update(b.dfs[r.target_table].columns.values)
    if segmentation_name in colum_names:
        raise ValueError(f"{segmentation_name} already exists")

    if edge_direction == EdgeDirection.Ignore:
        graph = graph.to_undirected()
        components = nx.connected_components(graph)
    else:
        components = nx.strongly_connected_components(graph)

    for table in b.dfs:
        b.dfs[table] = b.dfs[table].copy()

    mapping = {}
    table_id_cols = {}

    for comp_id, comp in enumerate(components):
        for node in comp:
            m = meta[node]
            mapping[str(m.node_id)] = comp_id
            table_id_cols[m.table] = m.id_column

    for table, id_column in table_id_cols.items():
        b.dfs[table][segmentation_name] = b.dfs[table][id_column].astype(str).map(mapping)
    return b
