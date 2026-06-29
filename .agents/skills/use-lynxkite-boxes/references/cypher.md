**Cypher:**
Run a Cypher query on the graph in the bundle. Save the results as a new DataFrame.
parameters:
  - query: typing.Annotated[str, {'format': 'textarea'}] = ? --?
  - save_as: <class 'str'> = results --?
  - bundle: <class 'lynxkite_graph_analytics.bundle.Bundle'> = ? --?

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.query_ops.cypher(query=<query_value>, save_as=<save_as_value>, bundle=<bundle_variable>)
