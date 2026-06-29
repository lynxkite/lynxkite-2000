**Scatter plot:**

parameters:
  - x: typing.Annotated[tuple[str, str], {'format': 'double-dropdown', 'metadata_query1': '[].dataframes[].keys(@)[]', 'metadata_query2': '[].dataframes[].<first>.columns[]'}] = ? --?
  - y: typing.Annotated[tuple[str, str], {'format': 'double-dropdown', 'metadata_query1': '[].dataframes[].keys(@)[]', 'metadata_query2': '[].dataframes[].<first>.columns[]'}] = ? --?
  - b: <class 'lynxkite_graph_analytics.bundle.Bundle'> = ? --?

returns:


usage:
output_variable = lynxkite_graph_analytics.operations.visualization_ops.scatter_plot(x=<x_value>, y=<y_value>, b=<b_variable>)
