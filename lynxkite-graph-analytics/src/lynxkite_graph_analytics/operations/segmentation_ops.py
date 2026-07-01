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

    column_names = set()
    for r in b.relations:
        column_names.update(b.dfs[r.source_table].columns.values)
        column_names.update(b.dfs[r.target_table].columns.values)
    if segmentation_name in column_names:
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
            "value": unique_values,
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


def _find_segmentation_relations(b, segmentation_name):
    """
    Finds all relations that connect nodes to a segmentation table.
    Returns a list of (relation, node_table) tuples.
    :param b: the bundle
    :param segmentation_name: the name of the segmentation
    """
    if segmentation_name not in b.dfs:
        raise ValueError(f"Segmentation table {segmentation_name} does not exist")

    relations = []
    for relation in b.relations:
        if relation.target_table == segmentation_name:
            relations.append((relation, relation.source_table))

    if not relations:
        raise ValueError(f"No relations found connecting to segmentation {segmentation_name}")

    return relations


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

    # Find relations connecting to the segmentation
    relations = _find_segmentation_relations(b, segmentation_name)

    columns = {item[0] for item in aggregations}
    agg_dict = {col: funcs.split(" ") for col, funcs in aggregations}
    _suffix_check(add_suffixes, agg_dict.values())

    all_tables = []
    for relation, node_table in relations:
        # Get the edge table
        edge_df = b.dfs[relation.df].copy()
        node_df = b.dfs[node_table]

        # Merge nodes with edges to get segment assignments
        merged = edge_df.merge(
            node_df[[relation.source_key] + list(columns)],
            left_on=relation.source_column,
            right_on=relation.source_key,
            how="left",
        )

        # Keep only the segment ID and the columns to aggregate
        merged = merged[[relation.target_column] + list(columns)]
        merged = merged.rename(columns={relation.target_column: segmentation_name})
        all_tables.append(merged)

    combined = pandas.concat(all_tables, ignore_index=True)

    # Group by segment and aggregate
    aggregated = combined.groupby(segmentation_name).agg(agg_dict)  # type: ignore
    aggregated.columns = [f"{col}_{func}" for col, func in aggregated.columns]
    aggregated = aggregated.reset_index()

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

    # Find relations connecting to the segmentation
    relations = _find_segmentation_relations(b, segmentation_name)

    columns = {item[0] for item in aggregations}
    agg_dict = {col: funcs.split(" ") for col, funcs in aggregations}
    _suffix_check(add_suffixes, agg_dict.values())

    aggregated = []
    for relation, node_table in relations:
        # Get the edge table and node table
        edge_df = b.dfs[relation.df]
        node_df = b.dfs[node_table]

        # Create a mapping from segment ID to list of node IDs
        segment_to_nodes = {}
        for idx, row in edge_df.iterrows():
            segment_id = row[relation.target_column]
            node_id = row[relation.source_column]
            if segment_id not in segment_to_nodes:
                segment_to_nodes[segment_id] = []
            segment_to_nodes[segment_id].append(node_id)

        # For each node, find nodes in the same segments and aggregate
        for idx, row in node_df.iterrows():
            node_id = row[relation.source_key]

            # Find all nodes that share a segment with this node
            node_segments = edge_df[edge_df[relation.source_column] == node_id][
                relation.target_column
            ].tolist()
            nodes_in_segments = set()
            for segment_id in node_segments:
                if segment_id in segment_to_nodes:
                    nodes_in_segments.update(segment_to_nodes[segment_id])

            # Get all nodes in the same segments
            relevant_nodes = node_df[node_df[relation.source_key].isin(nodes_in_segments)]

            # Aggregate
            row_aggregation = {
                "table_id": f"{node_table}_{node_id}",
                segmentation_name: set(node_segments),
            }

            agg_res = relevant_nodes[list(columns)].agg(agg_dict)
            if isinstance(agg_res, pandas.DataFrame):
                for col, func in agg_res.columns:
                    row_aggregation[f"{col}_{func}"] = agg_res.iloc[0][(col, func)]
            else:
                # If it's a Series, it means single aggregation per column
                for col, func in agg_res.index:
                    row_aggregation[f"{col}_{func}"] = agg_res[(col, func)]

            aggregated.append(row_aggregation)

    b.dfs["aggregated"] = pandas.DataFrame(aggregated)
    return b
