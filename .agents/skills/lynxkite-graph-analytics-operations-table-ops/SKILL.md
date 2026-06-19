---
name: lynxkite-graph-analytics-operations-table-ops
description: Collection of operations - Sample table, Filter with formula, Vector from attribute pair, Add rank attribute, Rename table, Select Table, Derive property, Enter table data, Join tables
---

**Sample table:**

parameters:
  - table_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}] = meta --?
  - fraction: <class 'float'> = 0.1 --?
  - b: <class 'lynxkite_graph_analytics.bundle.Bundle'> = ? --?

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.table_ops.sample_table(table_name=<table_name_value>, fraction=<fraction_value>, b=<b_variable>)

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

**Vector from attribute pair:**
Creates a new column with vectors that contain the two attributes
parameters:
  - table_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}] = ? --?
  - attribute1: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].<table_name>.columns[]'}] = ? --?
  - attribute2: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].<table_name>.columns[]'}] = ? --?
  - new_name: <class 'str'> = ? --?
  - b: <class 'lynxkite_graph_analytics.bundle.Bundle'> = ? --?

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.table_ops.vector_from_attribute_pair(table_name=<table_name_value>, attribute1=<attribute1_value>, attribute2=<attribute2_value>, new_name=<new_name_value>, b=<b_variable>)

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

**Select Table:**

parameters:
  - table_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}] = ? --?
  - b: <class 'lynxkite_graph_analytics.bundle.Bundle'> = ? --?

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.table_ops.select_table(table_name=<table_name_value>, b=<b_variable>)

**Derive property:**

parameters:
  - table_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}] = ? --?
  - formula: typing.Annotated[str, {'format': 'textarea'}] = ? --?
  - b: <class 'lynxkite_graph_analytics.bundle.Bundle'> = ? --?

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.table_ops.derive_property(table_name=<table_name_value>, formula=<formula_value>, b=<b_variable>)

**Enter table data:**
Enter table data as CSV. The first row should contain column names.
parameters:
  - table_name: <class 'str'> = ? --?
  - data: typing.Annotated[str, {'format': 'textarea'}] = ? --?

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.table_ops.enter_table_data(table_name=<table_name_value>, data=<data_value>)

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
