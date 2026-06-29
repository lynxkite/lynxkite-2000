**Rename table:**
Assigns a new name to the table
parameters:
  - old_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}] = ? --?
  - new_name: <class 'str'> = ? --?
  - b: <class 'lynxkite_graph_analytics.bundle.Bundle'> = ? --?

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.table_ops.rename_table(old_name=<old_name_value>, new_name=<new_name_value>, b=<b_variable>)
