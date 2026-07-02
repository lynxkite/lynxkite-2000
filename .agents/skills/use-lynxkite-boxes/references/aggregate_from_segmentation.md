**Aggregate from segmentation:**
For every node it aggregates the specified parameters of every node that shares a segment with it.
parameters:
  - relation_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].relations[].name'}] = ? --the relation connecting the node table to the segmentation table
  - add_suffixes: <class 'bool'> = ? --whether to add suffixes or not
  - aggregations: typing.Annotated[list[tuple[str, str]], {'format': 'double-textbox_adder', 'metadata_query1': '[].dataframes[].<table_name>.columns[]'}] = ? --the aggregations to perform, specified as a list of tuples (column_name, aggregation_function(https://pandas.pydata.org/pandas-docs/stable/reference/groupby.html#dataframegroupby-computations-descriptive-stats))
  - b: <class 'lynxkite_graph_analytics.bundle.Bundle'> = ? --the bundle to operate on

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.segmentation_ops.aggregate_from_segmentation(relation_name=<relation_name_value>, add_suffixes=<add_suffixes_value>, aggregations=<aggregations_value>, b=<b_variable>)
