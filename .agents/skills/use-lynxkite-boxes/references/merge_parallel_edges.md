**Merge parallel edges:**
Merges parallel edges, and aggregates the attributes with the specified functions(https://pandas.pydata.org/pandas-docs/stable/reference/groupby.html#dataframegroupby-computations-descriptive-stats).
parameters:
  - table_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}] = ? --the name of the table
  - source_key: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].<table_name>.columns[]'}] = ? --the name of the key in the source table
  - target_key: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].<table_name>.columns[]'}] = ? --the name of the key in the target table
  - aggregations: typing.Annotated[list[tuple[str, list[str]]], {'format': 'dropdown-multidropdown_adder', 'metadata_query1': '[].dataframes[].<table_name>.columns[]', 'options2': ['sum', 'mean', 'median', 'min', 'max', 'prod', 'std', 'var', 'sem', 'skew', 'count', 'size', 'first', 'last']}] = ? --the aggregations to perform, specified as a list of tuples
  - b: <class 'lynxkite_graph_analytics.bundle.Bundle'> = ? --the bundle

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.graph_ops.merge_parallel_edges(table_name=<table_name_value>, source_key=<source_key_value>, target_key=<target_key_value>, aggregations=<aggregations_value>, b=<b_variable>)
