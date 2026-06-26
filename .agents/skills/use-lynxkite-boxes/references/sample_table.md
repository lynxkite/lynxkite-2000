**Sample table:**

parameters:
  - table_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}] = meta --?
  - fraction: <class 'float'> = 0.1 --?
  - b: <class 'lynxkite_graph_analytics.bundle.Bundle'> = ? --?

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.table_ops.sample_table(table_name=<table_name_value>, fraction=<fraction_value>, b=<b_variable>)
