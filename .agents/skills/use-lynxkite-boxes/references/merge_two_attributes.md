**Merge two attributes:**
An attribute may not be defined everywhere. This operation uses the secondary attribute to fill in the values where the primary attribute is undefined. If both are undefined then the result is undefined too.
parameters:
  - table_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}] = ? --the name of the table
  - new_attribute: <class 'str'> = ? --the name of the new attribute
  - primary_attribute: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].<table_name>.columns[]'}] = ? --the primary attribute to use
  - secondary_attribute: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].<table_name>.columns[]'}] = ? --the secondary attribute to use
  - b: <class 'lynxkite_graph_analytics.bundle.Bundle'> = ? --the bundle

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.graph_ops.merge_two_attributes(table_name=<table_name_value>, new_attribute=<new_attribute_value>, primary_attribute=<primary_attribute_value>, secondary_attribute=<secondary_attribute_value>, b=<b_variable>)
