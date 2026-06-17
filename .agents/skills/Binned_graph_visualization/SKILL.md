
---
name: Binned_graph_visualization
description: Nodes binned together by x and y are aggregated into one node.
---

Nodes binned together by x and y are aggregated into one node.
Edges between bins are aggregated into one edge.

parameters:
  - b: core.Bundle = None
  - x_property: core.NodePropertyName = None
  - y_property: core.NodePropertyName = None
  - x_bins: Any = 5
  - y_bins: Any = 5
  - show_loops: bool = False

usage:
output_variable = lynxkite_graph_analytics.src.lynxkite_graph_analytics.operations.visualization_ops.binned_graph_visualization(b=<b_variable>, x_property=<x_property_value>, y_property=<y_property_value>, x_bins=<x_bins_value>, y_bins=<y_bins_value>, show_loops=<show_loops_value>)

Replace <*_variable> with the output of a previous operation, and <*_value> with a constant value.
