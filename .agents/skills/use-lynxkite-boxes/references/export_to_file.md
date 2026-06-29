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
