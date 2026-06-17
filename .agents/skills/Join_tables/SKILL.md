
---
name: Join_tables
description: Join/merge dataframes from two bundles.
---

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
  - bundle_a: core.Bundle = None
  - bundle_b: core.Bundle = None
  - table_a: core.TableName = None
  - table_b: core.TableName = None
  - join_type: JoinType = JoinType.inner
  - on_column: str =
  - left_on: str =
  - right_on: str =
  - suffixes: str = _a,_b

usage:
output_variable = lynxkite_graph_analytics.src.lynxkite_graph_analytics.operations.table_ops.join_tables(bundle_a=<bundle_a_variable>, bundle_b=<bundle_b_variable>, table_a=<table_a_value>, table_b=<table_b_value>, join_type=<join_type_value>, on_column=<on_column_value>, left_on=<left_on_value>, right_on=<right_on_value>, suffixes=<suffixes_value>)

Replace <*_variable> with the output of a previous operation, and <*_value> with a constant value.
