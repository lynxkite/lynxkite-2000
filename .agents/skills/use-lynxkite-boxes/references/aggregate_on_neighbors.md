**Aggregate on neighbors:**

parameters:
  - property: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].nodes[].columns[]'}] = ? --?
  - aggregation: <enum 'AggregationMethod'> = ? --?
  - g: <class 'networkx.classes.graph.Graph'> = ? --?

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.graph_ops.aggregate_on_neighbors(property=<property_value>, aggregation=<aggregation_value>, g=<g_variable>)
