
---
name: Bar_chart
description: Bar_chart
---



parameters:
  - b: core.Bundle = None
  - x: core.TableColumn = None
  - y: core.TableColumn = None

usage:
output_variable = lynxkite_graph_analytics.src.lynxkite_graph_analytics.operations.visualization_ops.bar_chart(b=<b_variable>, x=<x_value>, y=<y_value>)

Replace <*_variable> with the output of a previous operation, and <*_value> with a constant value.
