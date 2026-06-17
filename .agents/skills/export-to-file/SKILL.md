---
name: export-to-file
description: Exports a DataFrame to a file.
---

Exports a DataFrame to a file.

Args:
    bundle: The bundle containing the DataFrame to export.
    table_name: The name of the DataFrame in the bundle to export.
    filename: The name of the file to export to.
    file_format: The format of the file to export to. Defaults to CSV.

parameters:
  - bundle: core.Bundle = None
  - table_name: str = None
  - filename: ops.PathStr = None
  - file_format: FileFormat = FileFormat.csv

usage:
output_variable = lynxkite_graph_analytics.operations.file_ops.export_to_file(bundle=<bundle_variable>, table_name=<table_name_value>, filename=<filename_value>, file_format=<file_format_value>)

Replace <*_variable> with the output of a previous operation, and <*_value> with a constant value.
