**Derive with SQL:**
Derives a new column with a SQL expression and stores it in the same table.
parameters:
  - table_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}] = ? --the name of the table to derive the column in.
  - formula: typing.Annotated[str, {'format': 'textarea'}] = ? --the formula to derive the column with.
  - name: <class 'str'> = ? --the name of the derived column.
  - b: <class 'lynxkite_graph_analytics.bundle.Bundle'> = ? --the bundle.

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.table_ops.derive_with_sql(table_name=<table_name_value>, formula=<formula_value>, name=<name_value>, b=<b_variable>)
