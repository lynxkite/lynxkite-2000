**Fill attributes with default values:**
An attribute may not be defined everywhere. This operation sets the provided values for the rows of the specified attributes where they are not defined.
parameters:
  - table_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}] = ? --the table to operate on
  - adder: typing.Annotated[list[tuple[str, str]], {'format': 'dropdown-textbox_adder', 'metadata_query1': '[].dataframes[].<table_name>.columns[]'}] = ? --the attributes and the values to set
  - b: <class 'lynxkite_graph_analytics.bundle.Bundle'> = ? --the bundle

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.table_ops.fill_with_default(table_name=<table_name_value>, adder=<adder_value>, b=<b_variable>)
