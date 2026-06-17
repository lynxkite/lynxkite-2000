---
name: sample-graph
description: Takes a (preferably connected) subgraph.
---

Takes a (preferably connected) subgraph.

parameters:
  - graph: nx.Graph = None
  - nodes: int = 100

usage:
output_variable = lynxkite_graph_analytics.operations.graph_ops.sample_graph(graph=<graph_variable>, nodes=<nodes_value>)

Replace <*_variable> with the output of a previous operation, and <*_value> with a constant value.
