---
name: rename-table
description: Assigns a new name to the table
---

Assigns a new name to the table

parameters:
  - b: core.Bundle = None
  - old_name: core.TableName = None
  - new_name: str = None

usage:
output_variable = lynxkite_graph_analytics.operations.table_ops.rename_table(b=<b_variable>, old_name=<old_name_value>, new_name=<new_name_value>)

Replace <*_variable> with the output of a previous operation, and <*_value> with a constant value.
