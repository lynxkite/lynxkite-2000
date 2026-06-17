---
name: histogram
description: histogram
---



parameters:
  - b: core.Bundle = None
  - column: core.TableColumn = None
  - bins: int = 20

usage:
output_variable = lynxkite_graph_analytics.operations.visualization_ops.histogram(b=<b_variable>, column=<column_value>, bins=<bins_value>)

Replace <*_variable> with the output of a previous operation, and <*_value> with a constant value.
