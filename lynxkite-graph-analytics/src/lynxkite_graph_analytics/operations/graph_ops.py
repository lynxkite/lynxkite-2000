"""Operations for graphs."""

import json
import typing
from lynxkite_core import ops
from collections import deque
import pandas as pd
from .. import core, bundle
import networkx as nx
import enum
import numpy as np


op = ops.op_registration(core.ENV, "Graph operations")


@op("Merge", icon="link")
def merge(
    bundles: list[core.Bundle],
    *,
    merge_mode: bundle.BundleMergeMode = bundle.BundleMergeMode.must_be_unique,
):
    """Merge multiple inputs"""
    b = bundle.merge_bundles(bundles, merge_mode=merge_mode)
    return b


@op("Merge nodes on attribute", icon="link")
def merge_nodes(
    b: core.Bundle,
    *,
    table_name: core.TableName,
    attribute: core.ColumnNameByTableName,
    add_suffixes: bool = False,
    aggregations: core.DropdownTextAdderByTableName,
) -> core.Bundle:
    """Merges the nodes that have the same value for the given attribute.
    The aggregations parameter is a list of tuples (column_name, aggregation_function(https://pandas.pydata.org/pandas-docs/stable/reference/groupby.html#dataframegroupby-computations-descriptive-stats)) that specifies
    which other columns should be included in the new DataFrame and how to aggregate them.
    :param b: the bundle
    :param table_name: the name of the table
    :param attribute: the name of the attribute to merge on
    :param add_suffixes: whether to add suffixes to the aggregated columns
    :param aggregations: the aggregations to perform, specified as a list of tuples
    """
    b = b.copy()
    for table in b.dfs.keys():
        b.dfs[table] = b.dfs[table].copy()

    relations = b.relations.copy()
    b.relations = []
    for r in relations:
        b.relations.append(r.copy())

    old_df = b.dfs[table_name].copy()
    agg_dict = {}
    name_dict = {}

    for column, funcs in aggregations:
        if column not in agg_dict:
            agg_dict[column] = []
        funcs = funcs.split(" ")
        if len(funcs) > 1 and not add_suffixes:
            raise ValueError(
                "Adding suffixes is required when multiple aggregation functions are specified for a column."
            )
        for func in funcs:
            if func not in agg_dict[column]:
                agg_dict[column].append(func)
            name_dict[(column, func)] = f"{column}_{func}" if add_suffixes else column
    grouped_df = old_df.groupby(attribute).agg(agg_dict)
    grouped_df.columns = [name_dict.get(col) for col in grouped_df.columns]
    b.dfs[table_name] = grouped_df.reset_index()

    def _update_relation(r, suffix, column_attr, key_attr):
        new_column = attribute + suffix
        edge_column = getattr(r, column_attr)
        node_key = getattr(r, key_attr)
        b.dfs[r.df][new_column] = b.dfs[r.df][edge_column].map(
            old_df.set_index(node_key)[attribute]
        )
        setattr(r, column_attr, new_column)
        setattr(r, key_attr, attribute)

    for r in b.relations:
        if table_name == r.source_table:
            _update_relation(r, "_src", "source_column", "source_key")
        if table_name == r.target_table:
            _update_relation(r, "_dst", "target_column", "target_key")

    return b


@op("Merge parallel edges", icon="link")
def merge_parallel_edges(
    b: core.Bundle,
    *,
    table_name: core.TableName,
    source_key: core.ColumnNameByTableName,
    target_key: core.ColumnNameByTableName,
    aggregations: core.DropdownTextAdderByTableName,
) -> core.Bundle:
    """
    Merges parallel edges, and aggregates the attributes with the specified functions(https://pandas.pydata.org/pandas-docs/stable/reference/groupby.html#dataframegroupby-computations-descriptive-stats).
    :param b: the bundle
    :param table_name: the name of the table
    :param source_key: the name of the key in the source table
    :param target_key: the name of the key in the target table
    :param aggregations: the aggregations to perform, specified as a list of tuples
    """
    b = b.copy()
    edges = b.dfs[table_name].copy()
    group_cols = [source_key, target_key]
    agg_dict = {}

    for column, funcs in aggregations:
        func_list = [f for f in funcs.split(" ") if f]
        if func_list:
            if column not in agg_dict:
                agg_dict[column] = []
            for func in func_list:
                if func not in agg_dict[column]:
                    agg_dict[column].append(func)

    if agg_dict:
        merged = edges.groupby(group_cols).agg(agg_dict).reset_index()
        new_columns = []
        for col, func in merged.columns:
            if func == "":
                new_columns.append(col)
            else:
                new_columns.append(f"{col}_{func}")

        merged.columns = new_columns
    else:
        merged = edges.drop_duplicates(subset=group_cols).reset_index(drop=True)

    b.dfs[table_name] = merged
    return b


@op("Define edges", view="graph_creation_view", outputs=["output"], icon="link")
def define_edges(b: core.Bundle, *, relations: str = ""):
    """Define edges between node tables"""
    b = b.copy()
    if relations.strip():
        new_relations = [core.RelationDefinition(**r) for r in json.loads(relations).values()]
        b.relations.extend(new_relations)
    return ops.Result(output=b, display=b.to_table_view(limit=100))


ColumnNameForSource = typing.Annotated[
    str, {"format": "dropdown", "metadata_query": "[].dataframes[].<source_table>.columns[]"}
]
ColumnNameForTarget = typing.Annotated[
    str, {"format": "dropdown", "metadata_query": "[].dataframes[].<target_table>.columns[]"}
]


@op("Connect nodes on attribute", icon="link")
def connect_nodes(
    b: core.Bundle,
    *,
    source_table: core.TableName,
    source_id: ColumnNameForSource,
    source_attribute: ColumnNameForSource,
    target_table: core.TableName,
    target_id: ColumnNameForTarget,
    target_attribute: ColumnNameForTarget,
) -> core.Bundle:
    """
    Creates edges between nodes from table1 and table2 if the two attributes of the node are equal.

    Parameters:
    - source_table: Name of the first table
    - source_id: ID column in the first table
    - source_attribute: Attribute column in the first table used for matching
    - target_table: Name of the second table
    - target_id: ID column in the second table
    - target_attribute: Attribute column in the second table used for matching
    """
    b = b.copy()
    for table in b.dfs.keys():
        b.dfs[table] = b.dfs[table].copy()

    df1 = (b.dfs[source_table]).add_suffix("_src")
    df2 = (b.dfs[target_table]).add_suffix("_dst")
    source_key, target_key = source_id, target_id
    source_attribute, target_attribute, source_id, target_id = (
        source_attribute + "_src",
        target_attribute + "_dst",
        source_id + "_src",
        target_id + "_dst",
    )
    edges = pd.merge(df1, df2, left_on=source_attribute, right_on=target_attribute)

    if source_table == target_table:
        edges[[source_id, target_id]] = np.sort(edges[[source_id, target_id]], axis=1)
        edges = edges[edges[source_id] != edges[target_id]].drop_duplicates()

    b.dfs["edges"] = edges

    b.relations.append(
        core.RelationDefinition(
            name="graph",
            df="edges",
            source_column=source_id,
            source_table=source_table,
            source_key=source_key,
            target_column=target_id,
            target_table=target_table,
            target_key=target_key,
        )
    )
    return b


@op("Discard loop edges", icon="filter-filled")
def discard_loop_edges(graph: nx.Graph):
    graph = graph.copy()
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph


@op("Discard loop edges in relation", icon="filter-filled")
def discard_loop_edges_in_relation(b: core.Bundle, *, relation: core.RelationName):
    """
    Discards loop edges in the specified relation.
    :param b: the bundle
    :param relation: the relation
    """
    b = b.copy()
    for r in b.relations:
        if r.name == relation:
            df = b.dfs[r.df].copy()
            b.dfs[r.df] = df[df[r.source_column] != df[r.target_column]]
            break
    return b


@op("Discard parallel edges", icon="filter-filled")
def discard_parallel_edges(graph: nx.Graph):
    return nx.DiGraph(graph)


@op("Sample graph", icon="filter-filled")
def sample_graph(graph: nx.Graph, *, nodes: int = 100):
    """Takes a (preferably connected) subgraph."""
    sample = set()
    to_expand = deque([next(graph.nodes.keys().__iter__())])
    while to_expand and len(sample) < nodes:
        node = to_expand.pop()
        for n in graph.neighbors(node):
            if n not in sample:
                sample.add(n)
                to_expand.append(n)
            if len(sample) == nodes:
                break
    return nx.Graph(graph.subgraph(sample))


@op("Graph from edge list", icon="topology-star-3")
def graph_from_edge_list(
    df: pd.DataFrame, *, source: core.RecordsColumn, target: core.RecordsColumn
) -> core.Bundle:
    b = core.Bundle()
    b.dfs["nodes"] = pd.DataFrame({"id": pd.concat([df[source], df[target]]).unique()})
    b.dfs["edges"] = df.rename(columns={source: "source", target: "target"})
    b.relations.append(
        core.RelationDefinition(
            name="graph",
            df="edges",
            source_column="source",
            source_table="nodes",
            source_key="id",
            target_column="target",
            target_table="nodes",
            target_key="id",
        )
    )
    return b


class DegreeType(enum.StrEnum):
    in_degree = "in-degree"
    out_degree = "out-degree"
    degree = "degree"


@op("Degree", icon="topology-star-3")
def degree(g: nx.Graph) -> nx.Graph:
    g = g.copy()
    nx.set_node_attributes(g, name="in_degree", values=dict(g.in_degree()))
    nx.set_node_attributes(g, name="out_degree", values=dict(g.out_degree()))
    nx.set_node_attributes(g, name="degree", values=dict(g.degree()))
    return g


class AggregationMethod(enum.StrEnum):
    sum = "sum"
    mean = "mean"
    max = "max"
    min = "min"

    def apply(self, values):
        if self == AggregationMethod.sum:
            return np.sum(values)
        elif self == AggregationMethod.mean:
            return np.mean(values)
        elif self == AggregationMethod.max:
            return np.max(values)
        elif self == AggregationMethod.min:
            return np.min(values)
        else:
            raise ValueError(f"Unsupported aggregation method: {self}")


@op("Aggregate on neighbors", icon="topology-star-3")
def aggregate_on_neighbors(
    g: nx.Graph, *, property: core.NodePropertyName, aggregation: AggregationMethod
) -> nx.Graph:
    g = g.copy()
    for node in g.nodes:
        neighbor_values = [g.nodes[neighbor].get(property, 0) for neighbor in g.neighbors(node)]
        if not neighbor_values:
            continue
        agg_value = aggregation.apply(neighbor_values)
        g.nodes[node][f"{property}_neighborhood_{aggregation}"] = agg_value
    return g
