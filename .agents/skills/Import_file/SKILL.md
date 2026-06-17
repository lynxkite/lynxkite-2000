
---
name: Import_file
description: Read the contents of the a file into a `Bundle`.
---

Read the contents of the a file into a `Bundle`.

Args:
    file_path: Path to the file to import.
    table_name: Name to use for identifying the table in the bundle.
    file_format: Format of the file. Has to be one of the values in the `FileFormat` enum.

Returns:
    Bundle: Bundle with a single table with the contents of the file.

parameters:
  - file_path: ops.PathStr = None
  - table_name: str = None
  - file_format: FileFormat = FileFormat.csv

usage:
output_variable = lynxkite_graph_analytics.src.lynxkite_graph_analytics.operations.file_ops.import_file(file_path=<file_path_value>, table_name=<table_name_value>, file_format=<file_format_value>)

Replace <*_variable> with the output of a previous operation, and <*_value> with a constant value.
