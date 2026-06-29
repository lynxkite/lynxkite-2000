**Aggregate from segmentation:**
For every node it aggregates the specified parameters of every node that share a segment with it.
parameters:
  - segmentation_name: <class 'str'> = ? --the name of the segmentation to check for shared segments
  - add_suffixes: <class 'bool'> = ? --whether to add suffixes or not
  - aggregations: typing.Annotated[list[tuple[str, str]], {'format': 'double-textbox_adder', 'metadata_query1': '[].dataframes[].<table_name>.columns[]'}] = ? --?
  - b: <class 'lynxkite_graph_analytics.bundle.Bundle'> = ? --?

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.segmentation_ops.aggregate_from_segmentation(segmentation_name=<segmentation_name_value>, add_suffixes=<add_suffixes_value>, aggregations=<aggregations_value>, b=<b_variable>)
