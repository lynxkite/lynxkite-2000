**Merge nodes on attribute:**
Merges the nodes that have the same value for the given attribute.
The aggregations parameter is a list of tuples (column_name, aggregation_function(https://pandas.pydata.org/pandas-docs/stable/reference/groupby.html#dataframegroupby-computations-descriptive-stats)) that specifies
which other columns should be included in the new DataFrame and how to aggregate them.
parameters:
  - table_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}] = ? --the name of the table
  - attribute: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].<table_name>.columns[]'}] = ? --the name of the attribute to merge on
  - add_suffixes: <class 'bool'> = ? --whether to add suffixes to the aggregated columns
  - aggregations: typing.Annotated[list[tuple[str, str]], {'format': 'dropdown-textbox_adder', 'metadata_query1': '[].dataframes[].<table_name>.columns[]'}] = ? --the aggregations to perform, specified as a list of tuples
  - b: <class 'lynxkite_graph_analytics.bundle.Bundle'> = ? --the bundle

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.graph_ops.merge_nodes(table_name=<table_name_value>, attribute=<attribute_value>, add_suffixes=<add_suffixes_value>, aggregations=<aggregations_value>, b=<b_variable>)
