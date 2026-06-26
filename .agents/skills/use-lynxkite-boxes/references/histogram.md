**Histogram:**

parameters:
  - column: typing.Annotated[tuple[str, str], {'format': 'double-dropdown', 'metadata_query1': '[].dataframes[].keys(@)[]', 'metadata_query2': '[].dataframes[].<first>.columns[]'}] = ? --?
  - bins: <class 'int'> = 20 --?
  - b: <class 'lynxkite_graph_analytics.bundle.Bundle'> = ? --?

returns:


usage:
output_variable = lynxkite_graph_analytics.operations.visualization_ops.histogram(column=<column_value>, bins=<bins_value>, b=<b_variable>)
