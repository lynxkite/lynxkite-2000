---
name: select-table
description: select-table
---



parameters:
  - b: core.Bundle = None
  - table_name: core.TableName = None

usage:
output_variable = lynxkite_graph_analytics.operations.table_ops.select_table(b=<b_variable>, table_name=<table_name_value>)

Replace <*_variable> with the output of a previous operation, and <*_value> with a constant value.
