**Connect nodes on attribute:**
Creates edges between nodes from table1 and table2 if the two attributes of the node are equal.

Parameters:
- source_table: Name of the first table
- source_id: ID column in the first table
- source_attribute: Attribute column in the first table used for matching
- target_table: Name of the second table
- target_id: ID column in the second table
- target_attribute: Attribute column in the second table used for matching
parameters:
  - source_table: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}] = ? --?
  - source_id: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].<source_table>.columns[]'}] = ? --?
  - source_attribute: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].<source_table>.columns[]'}] = ? --?
  - target_table: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}] = ? --?
  - target_id: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].<target_table>.columns[]'}] = ? --?
  - target_attribute: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].<target_table>.columns[]'}] = ? --?
  - b: <class 'lynxkite_graph_analytics.bundle.Bundle'> = ? --?

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.graph_ops.connect_nodes(source_table=<source_table_value>, source_id=<source_id_value>, source_attribute=<source_attribute_value>, target_table=<target_table_value>, target_id=<target_id_value>, target_attribute=<target_attribute_value>, b=<b_variable>)
