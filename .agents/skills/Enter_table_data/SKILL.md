
---
name: Enter_table_data
description: Enter table data as CSV. The first row should contain column names.
---

Enter table data as CSV. The first row should contain column names.

parameters:
  - table_name: str = None
  - data: ops.LongStr = None

usage:
output_variable = lynxkite_graph_analytics.src.lynxkite_graph_analytics.operations.table_ops.enter_table_data(table_name=<table_name_value>, data=<data_value>)

Replace <*_variable> with the output of a previous operation, and <*_value> with a constant value.
