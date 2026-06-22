---
name: lynxkite-graph-analytics-operations-visualization-ops
description: Collection of operations - Visualize graph, Scatter plot, Bar chart, Histogram, Binned graph visualization
---

**Visualize graph:**

parameters:
  - color_nodes_by: typing.Optional[typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].nodes[].columns[]'}]] = ? --?
  - label_by: typing.Optional[typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].nodes[].columns[]'}]] = ? --?
  - color_edges_by: typing.Optional[typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].edges[].columns[]'}]] = ? --?
  - graph: <class 'lynxkite_graph_analytics.bundle.Bundle'> = ? --?

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.visualization_ops.visualize_graph(color_nodes_by=<color_nodes_by_value>, label_by=<label_by_value>, color_edges_by=<color_edges_by_value>, graph=<graph_variable>)

**Scatter plot:**

parameters:
  - x: typing.Annotated[tuple[str, str], {'format': 'double-dropdown', 'metadata_query1': '[].dataframes[].keys(@)[]', 'metadata_query2': '[].dataframes[].<first>.columns[]'}] = ? --?
  - y: typing.Annotated[tuple[str, str], {'format': 'double-dropdown', 'metadata_query1': '[].dataframes[].keys(@)[]', 'metadata_query2': '[].dataframes[].<first>.columns[]'}] = ? --?
  - b: <class 'lynxkite_graph_analytics.bundle.Bundle'> = ? --?

returns:


usage:
output_variable = lynxkite_graph_analytics.operations.visualization_ops.scatter_plot(x=<x_value>, y=<y_value>, b=<b_variable>)

**Bar chart:**

parameters:
  - x: typing.Annotated[tuple[str, str], {'format': 'double-dropdown', 'metadata_query1': '[].dataframes[].keys(@)[]', 'metadata_query2': '[].dataframes[].<first>.columns[]'}] = ? --?
  - y: typing.Annotated[tuple[str, str], {'format': 'double-dropdown', 'metadata_query1': '[].dataframes[].keys(@)[]', 'metadata_query2': '[].dataframes[].<first>.columns[]'}] = ? --?
  - b: <class 'lynxkite_graph_analytics.bundle.Bundle'> = ? --?

returns:


usage:
output_variable = lynxkite_graph_analytics.operations.visualization_ops.bar_chart(x=<x_value>, y=<y_value>, b=<b_variable>)

**Histogram:**

parameters:
  - column: typing.Annotated[tuple[str, str], {'format': 'double-dropdown', 'metadata_query1': '[].dataframes[].keys(@)[]', 'metadata_query2': '[].dataframes[].<first>.columns[]'}] = ? --?
  - bins: <class 'int'> = 20 --?
  - b: <class 'lynxkite_graph_analytics.bundle.Bundle'> = ? --?

returns:


usage:
output_variable = lynxkite_graph_analytics.operations.visualization_ops.histogram(column=<column_value>, bins=<bins_value>, b=<b_variable>)

**Binned graph visualization:**
Nodes binned together by x and y are aggregated into one node.
Edges between bins are aggregated into one edge.
parameters:
  - x_property: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].nodes[].columns[]'}] = ? --?
  - y_property: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].nodes[].columns[]'}] = ? --?
  - x_bins: <class 'int'> = 5 --?
  - y_bins: <class 'int'> = 5 --?
  - show_loops: <class 'bool'> = ? --?
  - b: <class 'lynxkite_graph_analytics.bundle.Bundle'> = ? --?

returns:


usage:
output_variable = lynxkite_graph_analytics.operations.visualization_ops.binned_graph_visualization(x_property=<x_property_value>, y_property=<y_property_value>, x_bins=<x_bins_value>, y_bins=<y_bins_value>, show_loops=<show_loops_value>, b=<b_variable>)
