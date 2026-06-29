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
