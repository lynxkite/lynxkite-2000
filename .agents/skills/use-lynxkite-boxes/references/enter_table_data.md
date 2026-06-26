**Enter table data:**
Enter table data as CSV. The first row should contain column names.
parameters:
  - table_name: <class 'str'> = ? --?
  - data: typing.Annotated[str, {'format': 'textarea'}] = ? --?

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.table_ops.enter_table_data(table_name=<table_name_value>, data=<data_value>)
