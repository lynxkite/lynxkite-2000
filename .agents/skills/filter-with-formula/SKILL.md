---
name: filter-with-formula
description: Removes all rows where the formula(https://numexpr.readthedocs.io/en/latest/user_guide.html#supported-functions) evaluates to false
---

Removes all rows where the formula(https://numexpr.readthedocs.io/en/latest/user_guide.html#supported-functions) evaluates to false

parameters:
  - b: core.Bundle = None
  - table_name: core.TableName = None
  - formula: ops.LongStr = None

usage:
output_variable = lynxkite_graph_analytics.operations.table_ops.filter_with_formula(b=<b_variable>, table_name=<table_name_value>, formula=<formula_value>)

Replace <*_variable> with the output of a previous operation, and <*_value> with a constant value.
