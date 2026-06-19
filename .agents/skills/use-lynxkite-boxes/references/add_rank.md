**Add rank attribute:**
Sorts the rows by the given attribute in the given order and creates a new column with the rank of the row
parameters:
  - table_column: typing.Annotated[tuple[str, str], {'format': 'double-dropdown', 'metadata_query1': '[].dataframes[].keys(@)[]', 'metadata_query2': '[].dataframes[].<first>.columns[]'}] = ? --The table and column to rank
  - rank_name: <class 'str'> = ? --The name of the new rank column
  - order: <enum 'OrderType'> = ? --The order in which to rank the rows either 'ascending' or 'descending'
  - b: <class 'lynxkite_graph_analytics.bundle.Bundle'> = ? --?

returns:
  - output: ? - The updated bundle with the new rank column.

usage:
output_variable = lynxkite_graph_analytics.operations.table_ops.add_rank(table_column=<table_column_value>, rank_name=<rank_name_value>, order=<order_value>, b=<b_variable>)
