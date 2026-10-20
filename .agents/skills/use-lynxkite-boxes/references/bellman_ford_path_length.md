**Bellman–Ford path length:**
Returns the shortest path length from source to target
in a weighted graph.
parameters:
  - source: <class 'str'> = ? --starting node for path
  - target: <class 'str'> = ? --ending node for path
  - weight: <class 'str'> = weight --If this is a string, then edge weights will be accessed via the
edge attribute with this key (that is, the weight of the edge
joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
such edge attribute exists, the weight of the edge is assumed to
be one.

If this is a function, the weight of an edge is the value
returned by the function. The function must accept exactly three
positional arguments: the two endpoints of an edge and the
dictionary of edge attributes for that edge. The function must
return a number.
  - G: <class 'networkx.classes.graph.Graph'> = ? --?
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.shortest_paths.weighted.bellman_ford_path_length(source=<source_value>, target=<target_value>, weight=<weight_value>, G=<G_variable>)
