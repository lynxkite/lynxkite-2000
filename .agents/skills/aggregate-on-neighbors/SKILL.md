---
name: aggregate-on-neighbors
description: aggregate-on-neighbors
---



parameters:
  - g: nx.Graph = None
  - property: core.NodePropertyName = None
  - aggregation: AggregationMethod = None

usage:
output_variable = lynxkite_graph_analytics.operations.graph_ops.aggregate_on_neighbors(g=<g_variable>, property=<property_value>, aggregation=<aggregation_value>)

Replace <*_variable> with the output of a previous operation, and <*_value> with a constant value.
