
---
name: Derive_property
description: Derive_property
---



parameters:
  - b: core.Bundle = None
  - table_name: core.TableName = None
  - formula: ops.LongStr = None

usage:
output_variable = lynxkite_graph_analytics.src.lynxkite_graph_analytics.operations.table_ops.derive_property(b=<b_variable>, table_name=<table_name_value>, formula=<formula_value>)

Replace <*_variable> with the output of a previous operation, and <*_value> with a constant value.
