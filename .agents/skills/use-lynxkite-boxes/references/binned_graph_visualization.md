**Binned graph visualization:**
Nodes binned together by x and y are aggregated into one node.
Edges between bins are aggregated into one edge.
parameters:
  - x_property: <class 'str'> = ? --?
  - y_property: <class 'str'> = ? --?
  - x_bins: <class 'int'> = 5 --?
  - y_bins: <class 'int'> = 5 --?
  - show_loops: <class 'bool'> = ? --?
  - b: <class 'lynxkite_graph_analytics.bundle.Bundle'> = ? --?

returns:


usage:
output_variable = lynxkite_graph_analytics.operations.visualization_ops.binned_graph_visualization(x_property=<x_property_value>, y_property=<y_property_value>, x_bins=<x_bins_value>, y_bins=<y_bins_value>, show_loops=<show_loops_value>, b=<b_variable>)
