**SQL:**
Run a SQL query on the DataFrames in the bundle. Save the results as a new DataFrame.
parameters:
  - query: typing.Annotated[str, {'format': 'textarea'}] = ? --?
  - save_as: <class 'str'> = results --?
  - bundle: <class 'lynxkite_graph_analytics.bundle.Bundle'> = ? --?

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.query_ops.sql(query=<query_value>, save_as=<save_as_value>, bundle=<bundle_variable>)
