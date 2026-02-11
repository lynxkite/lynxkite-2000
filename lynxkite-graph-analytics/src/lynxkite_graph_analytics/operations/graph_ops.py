"""Operations for graphs."""

from lynxkite_core import ops
from collections import deque
import pandas as pd
from .. import core
import networkx as nx
import enum
import numpy as np


op = ops.op_registration(core.ENV)


@op("Discard loop edges", icon="filter-filled")
def discard_loop_edges(graph: nx.Graph):
    graph = graph.copy()
    graph.remove_edges_from(nx.selfloop_edges(graph))
    return graph


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


@op("Graph from edge list", color="green", icon="topology-star-3")
def graph_from_edge_list(
    df: pd.DataFrame, *, source: core.DataFrameColumn, target: core.DataFrameColumn
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


@op("NetworkX", "Degree", icon="topology-star-3")
def degree(g: nx.Graph) -> nx.Graph:
    g = g.copy()
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
