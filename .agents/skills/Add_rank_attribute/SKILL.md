
---
name: Add_rank_attribute
description: Sorts the rows by the given attribute in the given order and creates a new column with the rank of the row
---

Sorts the rows by the given attribute in the given order and creates a new column with the rank of the row

parameters:
  - b: core.Bundle = None
  - table_column: core.TableColumn = None
  - rank_name: str = None
  - order: OrderType = None

usage:
output_variable = lynxkite_graph_analytics.src.lynxkite_graph_analytics.operations.table_ops.add_rank(b=<b_variable>, table_column=<table_column_value>, rank_name=<rank_name_value>, order=<order_value>)

Replace <*_variable> with the output of a previous operation, and <*_value> with a constant value.
