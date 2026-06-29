**Filter tables:**
Keeps/removes the selected tables based on the value of drop_selected
parameters:
  - drop_selected: <class 'bool'> = ? --if True, removes the selected tables, otherwise keeps them
  - tables: typing.Annotated[list[str], {'format': 'multi-dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}] = ? --the tables to keep/remove
  - b: <class 'lynxkite_graph_analytics.bundle.Bundle'> = ? --the bundle

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.table_ops.filter_tables(drop_selected=<drop_selected_value>, tables=<tables_value>, b=<b_variable>)
