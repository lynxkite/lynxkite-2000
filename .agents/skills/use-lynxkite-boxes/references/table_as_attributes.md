**Use table as attributes:**
Uses the columns from one table as attributes for the other.
parameters:
  - table_id: typing.Annotated[tuple[str, str], {'format': 'double-dropdown', 'metadata_query1': '[].dataframes[].keys(@)[]', 'metadata_query2': '[].dataframes[].<first>.columns[]'}] = ? --the table that gets the attributes
  - attribute_table_id: typing.Annotated[tuple[str, str], {'format': 'double-dropdown', 'metadata_query1': '[].dataframes[].keys(@)[]', 'metadata_query2': '[].dataframes[].<first>.columns[]'}] = ? --the table that provides the attributes
  - merge_mode: <enum 'MergeMode'> = ? --determines what happens if an attribute already exists in the original table.
Merge, prefer the table’s version: Where the table defines new values, those will be used. Elsewhere the existing values are kept.
Merge, prefer the graph’s version: Where the vertex attribute is already defined, it is left unchanged. Elsewhere the value from the table is used.
Merge, report error on conflict: An assertion is made to ensure that the values in the table are identical to the values in the graph on vertices where both are defined.
Keep the graph’s version: The data in the table is ignored.
Use the table’s version: The attribute is deleted from the graph and replaced with the attribute imported from the table.
Disallow this: A name conflict is treated as an error.
  - bundle_graph: <class 'lynxkite_graph_analytics.bundle.Bundle'> = ? --the bundle of the graph
  - bundle_att: <class 'lynxkite_graph_analytics.bundle.Bundle'> = ? --the bundle of the attributes

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.graph_ops.table_as_attributes(table_id=<table_id_value>, attribute_table_id=<attribute_table_id_value>, merge_mode=<merge_mode_value>, bundle_graph=<bundle_graph_variable>, bundle_att=<bundle_att_variable>)
