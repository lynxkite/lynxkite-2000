---
name: visualize-graph
description: visualize-graph
---



parameters:
  - graph: core.Bundle = None
  - color_nodes_by: core.NodePropertyName =
  - label_by: core.NodePropertyName =
  - color_edges_by: core.EdgePropertyName =
  - pick_nodes_by: core.NodePropertyName =
  - equals: Any = None
  - hops: int = 0

usage:
output_variable = lynxkite_graph_analytics.operations.visualization_ops.visualize_graph(graph=<graph_variable>, color_nodes_by=<color_nodes_by_value>, label_by=<label_by_value>, color_edges_by=<color_edges_by_value>, pick_nodes_by=<pick_nodes_by_value>, equals=<equals_value>, hops=<hops_value>)

Replace <*_variable> with the output of a previous operation, and <*_value> with a constant value.
