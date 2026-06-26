**Join tables:**
Join/merge dataframes from two bundles.

Parameters:
- table_a: Table name from bundle A
- table_b: Table name from bundle B
- join_type: Type of join - "inner", "outer", "left", "right", "cross"
- on_column: Column name to join on (same name in both tables)
- left_on: Column name in left table (when column names differ)
- right_on: Column name in right table (when column names differ)
- suffixes: Suffixes for overlapping columns (comma-separated, e.g., "_a,_b")
parameters:
  - table_a: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}] = ? --?
  - table_b: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}] = ? --?
  - join_type: <enum 'JoinType'> = inner --?
  - on_column: <class 'str'> = ? --?
  - left_on: <class 'str'> = ? --?
  - right_on: <class 'str'> = ? --?
  - suffixes: <class 'str'> = _a,_b --?
  - bundle_a: <class 'lynxkite_graph_analytics.bundle.Bundle'> = ? --?
  - bundle_b: <class 'lynxkite_graph_analytics.bundle.Bundle'> = ? --?

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.table_ops.join_tables(table_a=<table_a_value>, table_b=<table_b_value>, join_type=<join_type_value>, on_column=<on_column_value>, left_on=<left_on_value>, right_on=<right_on_value>, suffixes=<suffixes_value>, bundle_a=<bundle_a_variable>, bundle_b=<bundle_b_variable>)
