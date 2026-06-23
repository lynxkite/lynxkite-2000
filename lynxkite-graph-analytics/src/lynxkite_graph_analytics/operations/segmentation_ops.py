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
def connected_components(b: core.Bundle, *, edge_direction: EdgeDirection, segmentation_name: str):
    b = b.copy()

    graph, meta = b.to_nx_meta()

    colum_names = set()
    for r in b.relations:
        colum_names.update(b.dfs[r.source_table].columns.values)
        colum_names.update(b.dfs[r.target_table].columns.values)
    if segmentation_name in colum_names:
        raise ValueError(f"{segmentation_name} already exists")

    if edge_direction == EdgeDirection.Ignore:
        components = nx.connected_components(graph.to_undirected())
    else:
        components = nx.strongly_connected_components(graph)

    mapping = {}
    table_id_cols = {}

    for comp_id, comp in enumerate(list(components)):
        for node in comp:
            m = meta[node]
            mapping[m.table] = mapping.get(m.table, {})
            mapping[m.table][str(m.node_id)] = comp_id
            table_id_cols[m.table] = m.id_column

    for table, id_column in table_id_cols.items():
        b.dfs[table] = b.dfs[table].copy()
        b.dfs[table][segmentation_name] = b.dfs[table][id_column].astype(str).map(mapping[table])
    return b


@op("Segment by attribute", icon="filter-filled")
def segment_by_attribute(b: core.Bundle, *, attribute: str, segmentation_name: str):
    b = b.copy()

    node_tables = set()
    for r in b.relations:
        node_tables.add(r.target_table)
        node_tables.add(r.source_table)

    if segmentation_name in set.union(*[set(b.dfs[table].columns) for table in node_tables]):
        raise ValueError(f"{segmentation_name} already exists")
    if attribute not in set.intersection(*[set(b.dfs[table].columns) for table in node_tables]):
        raise ValueError(f"Every node has to have {attribute} attribute")

    values = set()
    for table in node_tables:
        values.update(b.dfs[table][attribute])

    mapping = {v: i for i, v in enumerate(values)}
    for table in node_tables:
        b.dfs[table] = b.dfs[table].copy()
        b.dfs[table][segmentation_name] = b.dfs[table][attribute].map(mapping)
    return b
