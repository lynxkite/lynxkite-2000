---
name: import-csv
description: Imports a CSV file.
---

Imports a CSV file.

parameters:
  - filename: ops.PathStr = None
  - columns: str = <from file>
  - separator: str = <auto>

usage:
output_variable = lynxkite_graph_analytics.operations.file_ops.import_csv(filename=<filename_value>, columns=<columns_value>, separator=<separator_value>)

Replace <*_variable> with the output of a previous operation, and <*_value> with a constant value.
