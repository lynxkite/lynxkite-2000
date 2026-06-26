**Graph from edge list:**

parameters:
  - source: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].records.columns[]'}] = ? --?
  - target: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].records.columns[]'}] = ? --?
  - df: <class 'pandas.core.frame.DataFrame'> = ? --?

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.graph_ops.graph_from_edge_list(source=<source_value>, target=<target_value>, df=<df_variable>)
