**Filter with formula:**
Removes all rows where the formula(https://numexpr.readthedocs.io/en/latest/user_guide.html#supported-functions) evaluates to false
parameters:
  - table_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}] = ? --?
  - formula: typing.Annotated[str, {'format': 'textarea'}] = ? --?
  - b: <class 'lynxkite_graph_analytics.bundle.Bundle'> = ? --?

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.table_ops.filter_with_formula(table_name=<table_name_value>, formula=<formula_value>, b=<b_variable>)
