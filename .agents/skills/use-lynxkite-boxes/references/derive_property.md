**Derive property:**

parameters:
  - table_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}] = ? --?
  - formula: typing.Annotated[str, {'format': 'textarea'}] = ? --?
  - b: <class 'lynxkite_graph_analytics.bundle.Bundle'> = ? --?

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.table_ops.derive_property(table_name=<table_name_value>, formula=<formula_value>, b=<b_variable>)
