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


@op("Find connected components", icon="filter-filled")
def connected_components(
    b: core.Bundle,
    *,
    relation_name: core.RelationName,
    edge_direction: EdgeDirection,
    segmentation_name: str,
):
    """
    Finds connected components in the graph of the relation.
    :param b: the bundle
    :param relation_name: the relation whose graph is segmented
    :param edge_direction: whether to ignore the direction of the edges
    :param segmentation_name: the name of the segmentation.
    """
    b = b.copy()
    for table in b.dfs.keys():
        b.dfs[table] = b.dfs[table].copy()

    relation = next((r for r in b.relations if r.name == relation_name))
    if relation.source_table != relation.target_table:
        raise ValueError("source_table and target_table must be the same")

    node_table = relation.source_table
    id_column = relation.source_key
    node_df = b.dfs[node_table]
    edge_df = b.dfs[relation.df]

    graph = nx.DiGraph()
    graph.add_nodes_from(node_df[id_column].tolist())
    graph.add_edges_from(
        zip(edge_df[relation.source_column].tolist(), edge_df[relation.target_column].tolist())
    )

    if edge_direction == EdgeDirection.Ignore:
        components = nx.connected_components(graph.to_undirected())
    else:
        components = nx.strongly_connected_components(graph)

    node_to_segment: dict[object, int] = {}
    segment_rows = []
    for comp_id, comp in enumerate(components):
        segment_rows.append({"id": comp_id, "value": comp_id})
        for node_id in comp:
            node_to_segment[node_id] = comp_id

    b.dfs[segmentation_name] = pandas.DataFrame(segment_rows)
    edge_table = f"{segmentation_name}_edges"
    b.dfs[edge_table] = pandas.DataFrame(
        {
            "node_id": node_df[id_column],
            "segment_id": node_df[id_column].map(node_to_segment),
        }
    )
    b.relations.append(
        core.RelationDefinition(
            name=f"{segmentation_name}_edges",
            df=edge_table,
            source_column="node_id",
            target_column="segment_id",
            source_table=node_table,
            target_table=segmentation_name,
            source_key=id_column,
            target_key="id",
        )
    )
    return b


@op("Segment by attribute", icon="category-2")
def segment_by_attribute(
    b: core.Bundle,
    *,
    table_name: core.TableName,
    attribute: core.ColumnNameByTableName,
    segmentation_name: str,
):
    """
    Segments the nodes in a table based on the values of the specified attribute.
    Creates a new table with segmentation IDs, edge table connecting nodes to segments,
    and a relation accordingly.
    :param b: the bundle
    :param table_name: the name of the table to segment
    :param attribute: the attribute to segment by
    :param segmentation_name: the name of the segmentation
    """
    b = b.copy()

    id_column: str | None = None
    for r in b.relations:
        if r.source_table == table_name:
            id_column = r.source_key
            break
        if r.target_table == table_name:
            id_column = r.target_key
            break

    if id_column is None:
        raise ValueError(f"{table_name} is not used in any relation")

    node_df = b.dfs[table_name]
    unique_values = node_df[attribute].unique()

    b.dfs[segmentation_name] = pandas.DataFrame(
        {
            "id": range(len(unique_values)),
            attribute: unique_values,
        }
    )

    edge_table_name = f"{segmentation_name}_edges"
    b.dfs[edge_table_name] = pandas.DataFrame(
        {
            "node_id": node_df[id_column],
            "segment_id": node_df[attribute].map({v: i for i, v in enumerate(unique_values)}),
        }
    )

    b.relations.append(
        core.RelationDefinition(
            name=f"{table_name}_{segmentation_name}",
            df=edge_table_name,
            source_column="node_id",
            target_column="segment_id",
            source_table=table_name,
            target_table=segmentation_name,
            source_key=id_column,
            target_key="id",
        )
    )

    return b


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
    relation_name: core.RelationName,
    add_suffixes: bool = False,
    aggregations: core.DoubleTextAdder,
):
    """
    For every segment in the segmentation it aggregates the specified parameters of the nodes belonging to it.
    :param b: the bundle to operate on
    :param relation_name: the relation connecting the node table to the segmentation table
    :param add_suffixes: whether to add suffixes or not
    :param aggregations: the aggregations to perform, specified as a list of tuples (column_name, aggregation_function(https://pandas.pydata.org/pandas-docs/stable/reference/groupby.html#dataframegroupby-computations-descriptive-stats))
    """
    b = b.copy()

    relation = next(r for r in b.relations if r.name == relation_name)

    segmentation_name = relation.target_table
    node_table = relation.source_table
    segment_df = b.dfs[segmentation_name].copy()

    parsed_aggregations = [(col, funcs.split(" ")) for col, funcs in aggregations]
    agg_dict = dict(parsed_aggregations)
    columns = [col for col, _ in parsed_aggregations]
    funcs = [func for _, func in parsed_aggregations]

    _suffix_check(add_suffixes, funcs)

    node_df = b.dfs[node_table][[relation.source_key] + columns].copy()
    merged = b.dfs[relation.df].merge(
        node_df, left_on=relation.source_column, right_on=relation.source_key, how="inner"
    )

    aggregated = merged.groupby(relation.target_column).agg(agg_dict)
    aggregated.columns = [
        f"{col}_{func}" if add_suffixes or len(agg_dict[col]) > 1 else col
        for col, func in aggregated.columns
    ]

    aggregated = aggregated.reset_index().rename(
        columns={relation.target_column: relation.target_key}
    )
    b.dfs[segmentation_name] = segment_df.merge(aggregated, on=relation.target_key, how="left")
    return b


@op("Aggregate from segmentation", icon="filter-filled")
def aggregate_from_segmentation(
    b: core.Bundle,
    *,
    relation_name: core.RelationName,
    add_suffixes: bool = False,
    aggregations: core.DoubleTextAdder,
):
    """
    For every node it aggregates the specified parameters of every node that shares a segment with it.
    :param b: the bundle to operate on
    :param relation_name: the relation connecting the node table to the segmentation table
    :param add_suffixes: whether to add suffixes or not
    :param aggregations: the aggregations to perform, specified as a list of tuples (column_name, aggregation_function(https://pandas.pydata.org/pandas-docs/stable/reference/groupby.html#dataframegroupby-computations-descriptive-stats))
    """
    b = b.copy()

    relation = next(r for r in b.relations if r.name == relation_name)

    node_table = relation.source_table
    node_df = b.dfs[node_table].copy()
    edge_df = b.dfs[relation.df].copy()

    parsed_aggregations = [(col, funcs.split(" ")) for col, funcs in aggregations]
    agg_dict = dict(parsed_aggregations)
    columns = [col for col, _ in parsed_aggregations]
    funcs = [func for _, func in parsed_aggregations]

    _suffix_check(add_suffixes, funcs)

    shared_segments = edge_df.merge(edge_df, on=relation.target_column, suffixes=("_a", "_b"))
    shared_segments = shared_segments[
        [f"{relation.source_column}_a", f"{relation.source_column}_b"]
    ].drop_duplicates()

    node_data = node_df[[relation.source_key] + columns]
    merged = shared_segments.merge(
        node_data,
        left_on=f"{relation.source_column}_b",
        right_on=relation.source_key,
        how="inner",
    )

    aggregated = merged.groupby(f"{relation.source_column}_a").agg(agg_dict)
    aggregated.columns = [
        f"{col}_{func}" if add_suffixes or len(agg_dict[col]) > 1 else col
        for col, func in aggregated.columns
    ]

    aggregated = aggregated.reset_index().rename(
        columns={f"{relation.source_column}_a": relation.source_key}
    )
    node_df = node_df[
        [
            col
            for col in node_df.columns
            if col not in aggregated.columns or col == relation.source_key
        ]
    ]
    b.dfs[node_table] = node_df.merge(aggregated, on=relation.source_key, how="left")
    return b
