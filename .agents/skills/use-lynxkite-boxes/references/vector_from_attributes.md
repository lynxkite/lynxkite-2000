**Vector from attributes:**
Creates a new column with vectors that contain the selected attributes in the selected order
parameters:
  - table_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}] = ? --?
  - attributes: typing.Annotated[list[str], {'format': 'multi-dropdown', 'metadata_query': '[].dataframes[].<table_name>.columns[]'}] = ? --?
  - vector_name: <class 'str'> = ? --?
  - b: <class 'lynxkite_graph_analytics.bundle.Bundle'> = ? --?

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.table_ops.vector_from_attributes(table_name=<table_name_value>, attributes=<attributes_value>, vector_name=<vector_name_value>, b=<b_variable>)
