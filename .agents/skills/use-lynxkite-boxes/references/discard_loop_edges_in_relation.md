**Discard loop edges in relation:**
Discards loop edges in the specified relation.
parameters:
  - relation: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].relations[].name'}] = ? --the relation
  - b: <class 'lynxkite_graph_analytics.bundle.Bundle'> = ? --the bundle

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.graph_ops.discard_loop_edges_in_relation(relation=<relation_value>, b=<b_variable>)
