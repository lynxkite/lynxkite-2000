---
name: lynxkite-graph-analytics-operations-file-ops
description: Collection of operations - Import file, Export to file, Import Parquet, Import CSV, Import GraphML, Graph from OSM
---

**Import file:**
Read the contents of the a file into a `Bundle`.
parameters:
  - file_path: typing.Annotated[str, {'format': 'path'}] = ? --Path to the file to import.
  - table_name: <class 'str'> = ? --Name to use for identifying the table in the bundle.
  - file_format: <enum 'FileFormat'> = csv --Format of the file. Has to be one of the values in the `FileFormat` enum.
  - file_format_group: group = csv --?

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.file_ops.import_file(file_path=<file_path_value>, table_name=<table_name_value>, file_format=<file_format_value>, file_format_group=<file_format_group_value>)

**Export to file:**
Exports a DataFrame to a file.
parameters:
  - table_name: <class 'str'> = ? --The name of the DataFrame in the bundle to export.
  - filename: typing.Annotated[str, {'format': 'path'}] = ? --The name of the file to export to.
  - file_format: <enum 'FileFormat'> = csv --The format of the file to export to. Defaults to CSV.
  - bundle: <class 'lynxkite_graph_analytics.bundle.Bundle'> = ? --The bundle containing the DataFrame to export.

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.file_ops.export_to_file(table_name=<table_name_value>, filename=<filename_value>, file_format=<file_format_value>, bundle=<bundle_variable>)

**Import Parquet:**
Imports a Parquet file.
parameters:
  - filename: typing.Annotated[str, {'format': 'path'}] = ? --?

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.file_ops.import_parquet(filename=<filename_value>)

**Import CSV:**
Imports a CSV file.
parameters:
  - filename: typing.Annotated[str, {'format': 'path'}] = ? --?
  - columns: <class 'str'> = <from file> --?
  - separator: <class 'str'> = <auto> --?

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.file_ops.import_csv(filename=<filename_value>, columns=<columns_value>, separator=<separator_value>)

**Import GraphML:**
Imports a GraphML file.
parameters:
  - filename: typing.Annotated[str, {'format': 'path'}] = ? --?

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.file_ops.import_graphml(filename=<filename_value>)

**Graph from OSM:**

parameters:
  - location: <class 'str'> = ? --?

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.file_ops.import_osm(location=<location_value>)
