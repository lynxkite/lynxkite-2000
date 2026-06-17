
---
name: Cypher
description: Run a Cypher query on the graph in the bundle. Save the results as a new DataFrame.
---

Run a Cypher query on the graph in the bundle. Save the results as a new DataFrame.

parameters:
  - bundle: core.Bundle = None
  - query: ops.LongStr = None
  - save_as: str = results

usage:
output_variable = lynxkite_graph_analytics.src.lynxkite_graph_analytics.operations.query_ops.cypher(bundle=<bundle_variable>, query=<query_value>, save_as=<save_as_value>)

Replace <*_variable> with the output of a previous operation, and <*_value> with a constant value.
