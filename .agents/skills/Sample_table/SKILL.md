
---
name: Sample_table
description: Sample_table
---



parameters:
  - b: core.Bundle = None
  - table_name: core.TableName = meta
  - fraction: float = 0.1

usage:
output_variable = lynxkite_graph_analytics.src.lynxkite_graph_analytics.operations.table_ops.sample_table(b=<b_variable>, table_name=<table_name_value>, fraction=<fraction_value>)

Replace <*_variable> with the output of a previous operation, and <*_value> with a constant value.
