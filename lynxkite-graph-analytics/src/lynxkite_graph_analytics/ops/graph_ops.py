"""Operations for graphs."""

from lynxkite_core import ops
from collections import deque

from .. import core
import networkx as nx


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
