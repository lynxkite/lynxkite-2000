"""Operations for segmentations."""

import enum
import networkx as nx
import pandas as pd

from lynxkite_core import ops
from .. import core

op = ops.op_registration(core.ENV, "Segmentation operations")


class EdgeDirection(enum.StrEnum):
    Ignore = "Ignore directions"
    Both = "Require both directions"


@op("Find connected components", icon="chart-dots-3")
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

    b.dfs[segmentation_name] = pd.DataFrame(segment_rows)
    edge_table = f"{segmentation_name}_edges"
    b.dfs[edge_table] = pd.DataFrame(
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

    b.dfs[segmentation_name] = pd.DataFrame(
        {
            "id": range(len(unique_values)),
            attribute: unique_values,
        }
    )

    edge_table_name = f"{segmentation_name}_edges"
    b.dfs[edge_table_name] = pd.DataFrame(
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


class Direction(enum.StrEnum):
    to_neighbour = "Aggregate to neighbour"
    from_neighbour = "Aggregate from neighbour"


@op("Aggregate between neighbours", icon="topology-star-3")
def aggregate_between_neighbours(
    b: core.Bundle,
    *,
    relation_name: core.RelationName,
    add_suffixes: bool,
    direction: Direction,
    aggregations: core.DoubleTextAdder,
) -> core.Bundle:
    """
    Depending on the direction, aggregates the specified columns nodes in one table to their neighbours in the other.
    :param b: the bundle to operate on
    :param relation_name: the relation connecting the two tables
    :param add_suffixes: whether to add suffixes or not
    :param direction: whether to aggregate "To" or "From" the target table
    :param aggregations: the aggregations to perform, specified as a list of tuples (column_name, aggregation_function)
    """
    b = b.copy()
    relation = next(r for r in b.relations if r.name == relation_name)

    parsed_aggregations = [(col, funcs.split(" ")) for col, funcs in aggregations]
    _suffix_check(add_suffixes, [funcs for _, funcs in parsed_aggregations])

    to_neighbour = direction == Direction.to_neighbour
    primary_pre = "target" if to_neighbour else "source"
    secondary_pre = "source" if to_neighbour else "target"

    primary_table = getattr(relation, f"{primary_pre}_table")
    primary_key = getattr(relation, f"{primary_pre}_key")
    primary_col = getattr(relation, f"{primary_pre}_column")
    secondary_table = getattr(relation, f"{secondary_pre}_table")
    secondary_key = getattr(relation, f"{secondary_pre}_key")
    secondary_col = getattr(relation, f"{secondary_pre}_column")

    cols = [col for col, _ in parsed_aggregations]
    secondary_df = b.dfs[secondary_table][[secondary_key] + cols].copy()
    merged = b.dfs[relation.df].merge(
        secondary_df, left_on=secondary_col, right_on=secondary_key, how="inner"
    )

    aggregated = merged.groupby(primary_col).agg(dict(parsed_aggregations))
    aggregated.columns = [
        f"{col}_{func}" if add_suffixes else col for col, func in aggregated.columns
    ]
    aggregated = aggregated.reset_index().rename(columns={primary_col: primary_key})

    primary_df = b.dfs[primary_table].copy()
    primary_df = primary_df[
        [col for col in primary_df.columns if col not in aggregated.columns or col == primary_key]
    ]

    b.dfs[primary_table] = primary_df.merge(aggregated, on=primary_key, how="left")
    return b
