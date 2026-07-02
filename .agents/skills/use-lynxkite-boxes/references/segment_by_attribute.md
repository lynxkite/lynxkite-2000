**Segment by attribute:**
Segments the nodes in a table based on the values of the specified attribute.
Creates a new table with segmentation IDs, edge table connecting nodes to segments,
and a relation accordingly.
parameters:
  - table_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}] = ? --the name of the table to segment
  - attribute: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].<table_name>.columns[]'}] = ? --the attribute to segment by
  - segmentation_name: <class 'str'> = ? --the name of the segmentation
  - b: <class 'lynxkite_graph_analytics.bundle.Bundle'> = ? --the bundle

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.segmentation_ops.segment_by_attribute(table_name=<table_name_value>, attribute=<attribute_value>, segmentation_name=<segmentation_name_value>, b=<b_variable>)
