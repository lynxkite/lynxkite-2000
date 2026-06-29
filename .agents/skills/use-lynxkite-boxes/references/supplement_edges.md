**Supplement edges with node attributes:**
Adds the attributes of the source and target nodes to the edges in the specified relation.
parameters:
  - table_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}] = ? --the name of the edge table
  - b: <class 'lynxkite_graph_analytics.bundle.Bundle'> = ? --the bundle

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.graph_ops.supplement_edges(table_name=<table_name_value>, b=<b_variable>)
