**Vector from attribute pair:**
Creates a new column with vectors that contain the two attributes
parameters:
  - table_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}] = ? --?
  - attribute1: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].<table_name>.columns[]'}] = ? --?
  - attribute2: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].<table_name>.columns[]'}] = ? --?
  - new_name: <class 'str'> = ? --?
  - b: <class 'lynxkite_graph_analytics.bundle.Bundle'> = ? --?

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.table_ops.vector_from_attribute_pair(table_name=<table_name_value>, attribute1=<attribute1_value>, attribute2=<attribute2_value>, new_name=<new_name_value>, b=<b_variable>)
