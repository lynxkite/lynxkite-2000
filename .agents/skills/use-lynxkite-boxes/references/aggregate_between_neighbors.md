**Aggregate between neighbors:**
Depending on the direction, aggregates the specified columns nodes in one table to their neighbors in the other.
parameters:
  - relation_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].relations[].name'}] = ? --the relation connecting the two tables
  - add_suffixes: <class 'bool'> = ? --whether to add suffixes or not
  - direction: <enum 'Direction'> = ? --whether to aggregate "To" or "From" the target table
  - aggregations: typing.Annotated[list[tuple[str, list[str]]], {'format': 'dropdown-multidropdown_relation_adder', 'directions': [<Direction.to_neighbor: 'Aggregate to neighbor'>, <Direction.from_neighbor: 'Aggregate from neighbor'>], 'options2': ['sum', 'mean', 'median', 'min', 'max', 'prod', 'std', 'var', 'sem', 'skew', 'count', 'size', 'first', 'last']}] = ? --the aggregations to perform, specified as a list of tuples (column_name, aggregation_function)
  - b: <class 'lynxkite_graph_analytics.bundle.Bundle'> = ? --the bundle to operate on

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.segmentation_ops.aggregate_between_neighbors(relation_name=<relation_name_value>, add_suffixes=<add_suffixes_value>, direction=<direction_value>, aggregations=<aggregations_value>, b=<b_variable>)
