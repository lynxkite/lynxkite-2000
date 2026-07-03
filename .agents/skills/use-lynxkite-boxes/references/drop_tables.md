**Drop tables:**
Keeps/removes the selected tables based on the value of drop_selected
parameters:
  - keep_selected: <class 'bool'> = ? --if False, removes the selected tables, otherwise the unselected ones
  - tables: typing.Annotated[list[str], {'format': 'multi-dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}] = ? --the tables to keep/remove
  - b: <class 'lynxkite_graph_analytics.bundle.Bundle'> = ? --the bundle

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.table_ops.drop_tables(keep_selected=<keep_selected_value>, tables=<tables_value>, b=<b_variable>)
