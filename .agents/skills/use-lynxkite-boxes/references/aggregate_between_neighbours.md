**Aggregate between neighbours:**
Depending on the direction, aggregates the specified columns nodes in one table to their neighbours in the other.
parameters:
  - relation_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].relations[].name'}] = ? --the relation connecting the two tables
  - add_suffixes: <class 'bool'> = ? --whether to add suffixes or not
  - direction: <enum 'Direction'> = ? --whether to aggregate "To" or "From" the target table
  - aggregations: typing.Annotated[list[tuple[str, str]], {'format': 'double-textbox_adder', 'metadata_query1': '[].dataframes[].<table_name>.columns[]'}] = ? --the aggregations to perform, specified as a list of tuples (column_name, aggregation_function)
  - b: <class 'lynxkite_graph_analytics.bundle.Bundle'> = ? --the bundle to operate on

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.segmentation_ops.aggregate_between_neighbours(relation_name=<relation_name_value>, add_suffixes=<add_suffixes_value>, direction=<direction_value>, aggregations=<aggregations_value>, b=<b_variable>)
