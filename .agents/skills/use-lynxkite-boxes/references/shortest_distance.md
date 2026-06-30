**Distance via shortest path:**
Computes the shortest distance from each node to the starting nodes using the specified edge distances.
parameters:
  - relation: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].relations[].name'}] = ? --the relation to use for the graph
  - edge_distances: <class 'str'> = ? --the distances for the edges
  - attribute_name: <class 'str'> = ? --the name of the attribute for storing the shortest distances
  - starting_distance: <class 'str'> = ? --the name of the attribute for the starting distances
  - max_iterations: <class 'str'> = ? --the maximum number of iterations allowed
  - undirected: <class 'bool'> = ? --whether to treat the graph as undirected or not
  - b: <class 'lynxkite_graph_analytics.bundle.Bundle'> = ? --the bundle

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.graph_ops.shortest_distance(relation=<relation_value>, edge_distances=<edge_distances_value>, attribute_name=<attribute_name_value>, starting_distance=<starting_distance_value>, max_iterations=<max_iterations_value>, undirected=<undirected_value>, b=<b_variable>)
