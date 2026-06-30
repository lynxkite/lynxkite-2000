**Aggregate to segmentation:**
For every segment in the segmentation it aggregates the specified parameters of the nodes belonging to it.
parameters:
  - segmentation_name: <class 'str'> = ? --the name of the segmentation
  - add_suffixes: <class 'bool'> = ? --whether to add suffixes or not
  - aggregations: typing.Annotated[list[tuple[str, str]], {'format': 'double-textbox_adder', 'metadata_query1': '[].dataframes[].<table_name>.columns[]'}] = ? --the aggregations to perform, specified as a list of tuples (column_name, aggregation_function(https://pandas.pydata.org/pandas-docs/stable/reference/groupby.html#dataframegroupby-computations-descriptive-stats))
  - b: <class 'lynxkite_graph_analytics.bundle.Bundle'> = ? --the bundle to operate on

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.segmentation_ops.aggregate_to_segmentation(segmentation_name=<segmentation_name_value>, add_suffixes=<add_suffixes_value>, aggregations=<aggregations_value>, b=<b_variable>)
