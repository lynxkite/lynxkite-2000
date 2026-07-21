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


@op("Merge", icon="arrows-join")
def merge(
    bundles: list[core.Bundle],
    *,
    merge_mode: bundle.BundleMergeMode = bundle.BundleMergeMode.must_be_unique,
):
    """Merge multiple inputs"""
    b = bundle.merge_bundles(bundles, merge_mode=merge_mode)
    return b


def get_id(b: core.Bundle, table_name: str) -> str:
    """Returns the id column of a table."""
    for relation in b.relations:
        if relation.source_table == table_name:
            return relation.source_key
        if relation.target_table == table_name:
            return relation.target_key
    raise ValueError(f"{table_name} is not used in any relation")


@op("Merge two attributes", icon="link")
def merge_two_attributes(
    b: core.Bundle,
    *,
    table_name: core.TableName,
    new_attribute: str,
    primary_attribute: core.ColumnNameByTableName,
    secondary_attribute: core.ColumnNameByTableName,
) -> core.Bundle:
    """
    An attribute may not be defined everywhere. This operation uses the secondary attribute to fill in the values where the primary attribute is undefined. If both are undefined then the result is undefined too.
    :param b: the bundle
    :param table_name: the name of the table
    :param new_attribute: the name of the new attribute
    :param primary_attribute: the primary attribute to use
    :param secondary_attribute: the secondary attribute to use
    """
    b = b.copy()
    df = b.dfs[table_name].copy()
    df[new_attribute] = df[primary_attribute].combine_first(df[secondary_attribute])
    b.dfs[table_name] = df
    return b


@op("Supplement edges with node attributes", icon="link")
def supplement_edges(b: core.Bundle, *, table_name: core.TableName) -> core.Bundle:
    """
    Adds the attributes of the source and target nodes to the edges in the specified relation.
    :param b: the bundle
    :param table_name: the name of the edge table
    """
    b = b.copy()
    for r in b.relations:
        if r.df == table_name:
            df = b.dfs[table_name].copy()
            source_df = b.dfs[r.source_table].copy()
            target_df = b.dfs[r.target_table].copy()
            for src_column in source_df.columns:
                if src_column != r.source_key:
                    df[f"{src_column}_src"] = df[r.source_column].map(
                        source_df.set_index(r.source_key)[src_column]
                    )
            for tgt_column in target_df.columns:
                if tgt_column != r.target_key:
                    df[f"{tgt_column}_dst"] = df[r.target_column].map(
                        target_df.set_index(r.target_key)[tgt_column]
                    )
            b.dfs[r.df] = df
    return b


class UnionFind:
    items: dict

    def __init__(self, items):
        self.items = {i: i for i in items}

    def find(self, i):
        path = []
        while self.items[i] != i:
            path.append(i)
            i = self.items[i]
        for node in path:
            self.items[node] = i
        return self.items[i]

    def union(self, a, b):
        self.items[b] = a


def _dual_growth(all_nodes, edges, prizes, super_root):
    uf = UnionFind(all_nodes)
    active = {c: (prizes[c] > 0 and c != super_root) for c in prizes}
    edge_slack = dict(edges)
    added_edges = []
    epsilon = 1e-12

    while any(active.values()):
        min_eps, event_edge, event_comp = float("inf"), None, None

        for (u, v), slack in edge_slack.items():
            cu, cv = uf.find(u), uf.find(v)
            if cu != cv:
                rate = active[cu] + active[cv]
                if rate > 0 and (slack / rate) < min_eps:
                    min_eps, event_edge, event_comp = slack / rate, (u, v), None

        for c, is_act in active.items():
            if is_act and prizes[c] < min_eps:
                min_eps, event_edge, event_comp = prizes[c], None, c

        if min_eps == float("inf") or (min_eps <= epsilon and not event_edge and not event_comp):
            break

        for c, is_act in active.items():
            if is_act:
                prizes[c] -= min_eps

        for u, v in list(edge_slack.keys()):
            cu, cv = uf.find(u), uf.find(v)
            if cu != cv:
                rate = active[cu] + active[cv]
                if rate > 0:
                    edge_slack[(u, v)] -= rate * min_eps

        if event_comp:
            active[event_comp] = False
        elif event_edge:
            u, v = event_edge
            cu, cv = uf.find(u), uf.find(v)
            added_edges.append((u, v))

            if cu == super_root or cv == super_root:
                old_c = cv if cu == super_root else cu
                uf.union(super_root, old_c)
                active[old_c] = False
            else:
                new_c, old_c = (cv, cu) if prizes[cv] >= prizes[cu] else (cu, cv)
                uf.union(new_c, old_c)
                prizes[new_c] += prizes[old_c]
                active[new_c] = prizes[new_c] > 0
                active[old_c] = False

    return added_edges


def _prune_tree(added_edges, edges, prizes, super_root):
    T = nx.Graph()
    for u, v in added_edges:
        T.add_edge(u, v, weight=edges[(u, v) if u < v else (v, u)])

    if super_root not in T:
        return T

    leaves = [n for n, d in T.degree() if n != super_root and d == 1]
    while leaves:
        node = leaves.pop()
        if node in T and T.degree(node) == 1 and node != super_root:
            nbr = next(T.neighbors(node))
            if prizes.get(node, 0.0) <= T[node][nbr]["weight"]:
                T.remove_node(node)
                if nbr != super_root and T.degree(nbr) == 1:
                    leaves.append(nbr)
    return T


def _gw_pcsf(nodes, und_list, node_prices, edge_costs, root_costs, eligible_root_nodes):
    if not nodes:
        return 0.0, set(), set(), set()

    super_root = "-1"
    edges = {e: float(edge_costs[e]) for e in und_list}
    for r in eligible_root_nodes:
        if r in nodes:
            edges[(super_root, r) if super_root < r else (r, super_root)] = float(
                root_costs.get(r, 0.0)
            )

    initial_prizes = {n: float(node_prices.get(n, 0.0)) for n in nodes}
    growth_prizes = dict(initial_prizes)
    growth_prizes[super_root] = float("inf")

    all_nodes = list(nodes) + [super_root]
    added_edges = _dual_growth(all_nodes, edges, growth_prizes, super_root)
    T = _prune_tree(added_edges, edges, initial_prizes, super_root)

    if super_root not in T or len(T) <= 1:
        return 0.0, set(), set(), set()

    sel_nodes = set(T.nodes) - {super_root}
    sel_roots = {r for r in eligible_root_nodes if T.has_edge(super_root, r)}
    sel_edges = {
        (u, v) if u < v else (v, u) for u, v in T.edges if u != super_root and v != super_root
    }

    profit = (
        sum(initial_prizes[n] for n in sel_nodes)
        - sum(edges[e] for e in sel_edges)
        - sum(float(root_costs.get(r, 0.0)) for r in sel_roots)
    )

    return max(0.0, float(profit)), sel_nodes, sel_roots, sel_edges


@op("Steiner forest", icon="eye", color="blue")
def pcsf(
    b: core.Bundle,
    *,
    relation: core.RelationName,
    price_column: str,
    weight_column: str,
    root_cost_column: str,
    output_edge: str,
    output_node: str,
    output_root_nodes: str,
    output_profit: str,
):
    """
    Creates a new dataframe that defines a Steiner Forest of the graph defined by the relation
    :param b: the bundle
    :param relation: the relation
    :param price_column: the column with the node prices
    :param weight_column: the column with the edge weights
    :param root_cost_column: the column with the root costs
    :param output_edge: the output column, 1.0 if the edge is part of the forest, None otherwise
    :param output_node: the output column, 1.0 if the node is part of the forest, None otherwise
    :param output_root_nodes: the output column, 1.0 if the node is a root node, None otherwise
    :param output_profit: a table with a single record: the profit

    """
    b = b.copy()
    rel = next((r for r in b.relations if r.name == relation))
    if rel.source_table != rel.target_table:
        raise ValueError("Source and target tables must be the same.")

    node_df, edge_df = b.dfs[rel.source_table].copy(), b.dfs[rel.df].copy()
    nid, src, dst = rel.source_key, rel.source_column, rel.target_column

    node_df[nid] = node_df[nid].astype(str)
    edge_df[[src, dst]] = edge_df[[src, dst]].astype(str)

    node_df[price_column] = pd.to_numeric(node_df[price_column]).fillna(0.0).clip(lower=0.0)
    edge_df[weight_column] = pd.to_numeric(edge_df[weight_column]).fillna(0.0).clip(lower=0.0)

    raw_root_costs = pd.to_numeric(node_df[root_cost_column])
    eligible_root_mask = raw_root_costs.notna() & (raw_root_costs >= 0)
    eligible_root_nodes = set(node_df.loc[eligible_root_mask, nid])
    node_df["root_cost_sanitized"] = raw_root_costs.fillna(0.0).clip(lower=0.0)

    nodes = list(node_df[nid].unique())
    node_prices = dict(zip(node_df[nid], node_df[price_column]))
    root_costs = dict(zip(node_df[nid], node_df["root_cost_sanitized"]))

    undirected_edges = {}
    edge_costs = {}

    for idx, row in edge_df.iterrows():
        u, v, w = row[src], row[dst], row[weight_column]
        if u == v or u not in node_prices or v not in node_prices:
            continue

        key = tuple(sorted([u, v]))
        if key not in edge_costs or w < edge_costs[key]:
            undirected_edges[key] = idx
            edge_costs[key] = w

    edges = list(undirected_edges.keys())

    net_profit, selected_nodes, selected_roots, selected_edges = _gw_pcsf(
        nodes=nodes,
        und_list=edges,
        node_prices=node_prices,
        edge_costs=edge_costs,
        root_costs=root_costs,
        eligible_root_nodes=eligible_root_nodes,
    )

    selected_edge_indices = {undirected_edges[e] for e in selected_edges if e in undirected_edges}

    node_df[output_node] = [1.0 if x in selected_nodes else None for x in node_df[nid]]
    node_df[output_root_nodes] = [1.0 if x in selected_roots else None for x in node_df[nid]]
    edge_df[output_edge] = [1.0 if idx in selected_edge_indices else None for idx in edge_df.index]

    results_df = pd.DataFrame(
        {output_profit: [float(net_profit) if net_profit is not None else None]}
    )
    b.dfs[output_profit] = results_df

    if "root_cost_sanitized" in node_df.columns:
        node_df.drop(columns=["root_cost_sanitized"], inplace=True)

    b.dfs[rel.source_table] = node_df.astype(object).where(node_df.notnull(), None)
    b.dfs[rel.df] = edge_df.astype(object).where(edge_df.notnull(), None)
    return b


def update_relations(
    b: core.Bundle,
    table_name: core.TableName,
    new_id: str,
    mapping: pd.Series,
) -> core.Bundle:
    """
    Updates the relations to use the new id column instead of the old ones.
    :param b: The bundle
    :param table_name: The name of the node table that was modified.
    :param new_id: The name of the new id attribute.
    :param mapping: Maps the old ids to the new ones.
    """
    b = b.copy()

    def _update_relation(r, suffix, column_attr, key_attr):
        new_column = new_id + suffix
        edge_column = getattr(r, column_attr)
        b.dfs[r.df][new_column] = b.dfs[r.df][edge_column].map(mapping)
        setattr(r, column_attr, new_column)
        setattr(r, key_attr, new_id)

    for r in b.relations:
        if table_name == r.source_table:
            _update_relation(r, "_src", "source_column", "source_key")
        if table_name == r.target_table:
            _update_relation(r, "_dst", "target_column", "target_key")

    return b


@op("Merge nodes on attribute", icon="affiliate")
def merge_nodes(
    b: core.Bundle,
    *,
    table_name: core.TableName,
    attribute: core.ColumnNameByTableName,
    add_suffixes: bool = False,
    aggregations: core.AggregationAdderByTableName,
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
        if len(funcs) > 1 and not add_suffixes:
            raise ValueError(
                "Adding suffixes is required when multiple aggregation functions are specified for a column."
            )
        for func in funcs:
            if func not in agg_dict[column]:
                agg_dict[column].append(func)
            name_dict[(column, func)] = f"{column}_{func}" if add_suffixes else column
    grouped_df = old_df.groupby(attribute).agg(agg_dict).replace({float("nan"): None})
    grouped_df.columns = [name_dict.get(col) for col in grouped_df.columns]
    b.dfs[table_name] = grouped_df.reset_index()
    update_relations(b, table_name, attribute, old_df.set_index(get_id(b, table_name))[attribute])
    return b


@op("Merge parallel edges", icon="arrows-right")
def merge_parallel_edges(
    b: core.Bundle,
    *,
    table_name: core.TableName,
    source_key: core.ColumnNameByTableName,
    target_key: core.ColumnNameByTableName,
    aggregations: core.AggregationAdderByTableName,
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
        func_list = [f for f in funcs if f]
        if func_list:
            if column not in agg_dict:
                agg_dict[column] = []
            for func in func_list:
                if func not in agg_dict[column]:
                    agg_dict[column].append(func)

    if agg_dict:
        merged = edges.groupby(group_cols).agg(agg_dict).replace({float("nan"): None}).reset_index()
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


@op("Distance via shortest path", icon="link")
def shortest_distance(
    b: core.Bundle,
    *,
    relation: core.RelationName,
    edge_distances: str,
    attribute_name: str,
    starting_distance: str,
    max_iterations: str,
    undirected: bool,
) -> core.Bundle:
    """
    Computes the shortest distance from each node to the starting nodes using the specified edge distances.
    :param b: the bundle
    :param relation: the relation to use for the graph
    :param edge_distances: the distances for the edges
    :param attribute_name: the name of the attribute for storing the shortest distances
    :param starting_distance: the name of the attribute for the starting distances
    :param max_iterations: the maximum number of iterations allowed
    :param undirected: whether to treat the graph as undirected or not
    """
    b = b.copy()

    for r in b.relations:
        if r.name == relation:
            edge_df = b.dfs[r.df].copy()
            source_table, source_col, target_col, source_key = (
                r.source_table,
                r.source_column,
                r.target_column,
                r.source_key,
            )
            break
    else:
        raise ValueError(f"Relation '{relation}' not found.")

    edge_df[source_col] = edge_df[source_col].astype(str).str.strip()
    edge_df[target_col] = edge_df[target_col].astype(str).str.strip()

    if undirected:
        reverse = edge_df.rename(columns={source_col: target_col, target_col: source_col})
        edge_df = pd.concat([edge_df, reverse], ignore_index=True)

    nodes = b.dfs[source_table].copy()
    nodes[source_key] = nodes[source_key].astype(str).str.strip()
    nodes = nodes.set_index(source_key, drop=False)
    nodes[attribute_name] = pd.to_numeric(nodes[starting_distance], errors="coerce")

    for _ in range(int(max_iterations)):
        current = nodes[attribute_name].dropna()
        if current.empty:
            break

        merged = edge_df.merge(current, left_on=source_col, right_index=True, how="inner")
        if merged.empty:
            break

        merged["_candidate"] = merged[attribute_name] + merged[edge_distances]
        best = merged.groupby(target_col)["_candidate"].min()

        before = nodes[attribute_name].copy()
        nodes = nodes.join(best.rename("_candidate"), how="left")
        nodes[attribute_name] = nodes[[attribute_name, "_candidate"]].min(axis=1, skipna=True)
        nodes.drop(columns="_candidate", inplace=True)

        if nodes[attribute_name].equals(before):
            break

    node_lookup = b.dfs[source_table][source_key].astype(str).str.strip()
    b.dfs[source_table][attribute_name] = node_lookup.map(nodes[attribute_name])

    return b


@op("Define edges", view="graph_creation_view", outputs=["output"], icon="route")
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


@op("Connect nodes on attribute", icon="share")
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
        edges = edges[edges[source_id] != edges[target_id]].drop_duplicates(
            subset=[source_id, target_id]
        )
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


@op("Discard loop edges in relation", icon="circle-off")
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
