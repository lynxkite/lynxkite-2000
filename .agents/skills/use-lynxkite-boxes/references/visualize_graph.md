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
