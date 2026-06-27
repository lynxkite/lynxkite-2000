"""Operations for tables."""

import enum
import networkx as nx
import pandas

from lynxkite_core import ops
from .. import core

op = ops.op_registration(core.ENV, "Segmentation operations")


class EdgeDirection(enum.StrEnum):
    Ignore = "Ignore directions"
    Both = "Require both directions"


@op("Find Connected Components", icon="filter-filled")
def connected_components(b: core.Bundle, *, edge_direction: EdgeDirection, segmentation_name: str):
    """
    Finds the connected components of the graph and put the nodes into a segment accordingly.
    :param b: the bundle
    :param edge_direction: whether to ignore the direction of the edges
    :param segmentation_name: the name of the segmentation.
    """
    b = b.copy()
    for table in b.dfs.keys():
        b.dfs[table] = b.dfs[table].copy()
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
            mapping[m.table][str(m.node_id)] = {comp_id}
            table_id_cols[m.table] = m.id_column

    for table, id_column in table_id_cols.items():
        b.dfs[table] = b.dfs[table].copy()
        b.dfs[table][segmentation_name] = b.dfs[table][id_column].astype(str).map(mapping[table])
    return b


@op("Segment by attribute", icon="filter-filled")
def segment_by_attribute(b: core.Bundle, *, attribute: str, segmentation_name: str):
    """
    Segments the nodes based on the values of the specified attribute.
    :param b: the bundle
    :param attribute: the attribute to segment by
    :param segmentation_name: the name of the segmentation
    """
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

    mapping = {v: {i} for i, v in enumerate(values)}
    for table in node_tables:
        b.dfs[table] = b.dfs[table].copy()
        b.dfs[table][segmentation_name] = b.dfs[table][attribute].map(mapping)
    return b


def _aggregation_prechecks(b, tables, columns, segmentation_name):
    """
    Checks whether the segmentation exists, and whether all columns exist in the tables with the segmentation.
    :param b: the bundle
    :param tables: the tables
    :param columns: the columns with the specified segmentation
    :param segmentation_name: the name of the segmentation
    """
    if len(tables) == 0:
        raise ValueError(f"{segmentation_name} does not exist")
    for table in tables:
        if not columns.issubset(b.dfs[table].columns):
            raise ValueError(f"Not all columns exist in table {table}")


def _suffix_check(add_suffixes, funcs_values):
    """
    Checks if suffixes are required for the aggregation functions.
    :param add_suffixes: whether to add suffixes or not
    :param funcs_values: the aggregation functions to check
    """
    for funcs in funcs_values:
        if len(funcs) > 1 and not add_suffixes:
            raise ValueError(
                "Adding suffixes is required when multiple aggregation functions are specified for a column."
            )


@op("Aggregate to segmentation", icon="filter-filled")
def aggregate_to_segmentation(
    b: core.Bundle,
    *,
    segmentation_name: str,
    add_suffixes: bool = False,
    aggregations: core.DoubleTextAdder,
):
    """
    For every segment in the segmentation it aggregates the specified parameters of the nodes belonging to it.
    :param b: the bundle to operate on
    :param segmentation_name: the name of the segmentation
    :param add_suffixes: whether to add suffixes or not
    :param aggregations: the aggregations to perform, specified as a list of tuples (column_name, aggregation_function(https://pandas.pydata.org/pandas-docs/stable/reference/groupby.html#dataframegroupby-computations-descriptive-stats))
    """
    b = b.copy()
    tables = [table for table in b.dfs.keys() if segmentation_name in b.dfs[table].columns]
    columns = {item[0] for item in aggregations}
    _aggregation_prechecks(b, tables, columns, segmentation_name)

    all_tables = []
    for table in tables:
        all_tables.append(b.dfs[table][list(columns) + [segmentation_name]])

    combined = pandas.concat(all_tables, ignore_index=True)
    combined = combined.explode(segmentation_name)
    agg_dict = {col: funcs.split(" ") for col, funcs in aggregations}
    _suffix_check(add_suffixes, agg_dict.values())

    aggregated = combined.groupby(segmentation_name).agg(agg_dict)
    aggregated.columns = [f"{col}_{func}" for col, func in aggregated.columns]
    aggregated = aggregated.reset_index()
    aggregated[segmentation_name] = aggregated[segmentation_name].apply(lambda x: {x})
    b.dfs["aggregated"] = aggregated
    return b


@op("Aggregate from segmentation", icon="filter-filled", slow=True)
def aggregate_from_segmentation(
    b: core.Bundle,
    *,
    segmentation_name: str,
    add_suffixes: bool = False,
    aggregations: core.DoubleTextAdder,
):
    """
    For every node it aggregates the specified parameters of every node that share a segment with it.
    :param segmentation_name: the name of the segmentation to check for shared segments
    :param add_suffixes: whether to add suffixes or not
    """
    b = b.copy()
    tables = [table for table in b.dfs.keys() if segmentation_name in b.dfs[table].columns]
    columns = {item[0] for item in aggregations}
    _aggregation_prechecks(b, tables, columns, segmentation_name)

    key_dict = {}
    for r in b.relations:
        key_dict[r.source_table] = r.source_key
        key_dict[r.target_table] = r.target_key

    all_tables = []
    for table in tables:
        df = b.dfs[table].copy()
        if table in key_dict.keys():
            df["_id"] = df[key_dict[table]]
        else:
            df["_id"] = df.index
        df["_table"] = table
        all_tables.append(df)

    combined = pandas.concat(all_tables, ignore_index=True)
    exploded = combined.explode(segmentation_name)
    agg_dict = {col: funcs.split(" ") for col, funcs in aggregations}
    _suffix_check(add_suffixes, agg_dict.values())

    aggregated = []
    for table in all_tables:
        for index, row in table.iterrows():
            relevant = exploded[exploded[segmentation_name].isin(row[segmentation_name])]
            row_aggregation = {
                "table_id": f"{row['_table']}_{row['_id']}",
                segmentation_name: row[segmentation_name],
            }
            unique_nodes = relevant.drop_duplicates(subset=["_id", "_table"])
            agg_res = unique_nodes.groupby(lambda x: "group").agg(agg_dict)
            for col, func in agg_res.columns:
                row_aggregation[f"{col}_{func}"] = agg_res.at["group", (col, func)]

            aggregated.append(row_aggregation)

    b.dfs["aggregated"] = pandas.DataFrame(aggregated)
    return b
