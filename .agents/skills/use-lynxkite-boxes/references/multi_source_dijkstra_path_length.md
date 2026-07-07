**Multi source Dijkstra path length:**
Find shortest weighted path lengths in G from a given set of
source nodes.

Compute the shortest path length between any of the source nodes and
all other reachable nodes for a weighted graph.
parameters:
  - cutoff: float | None = ? --Length (sum of edge weights) at which the search is stopped.
If cutoff is provided, only return paths with summed weight <= cutoff.
  - weight: <class 'str'> = weight --If this is a string, then edge weights will be accessed via the
edge attribute with this key (that is, the weight of the edge
joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
such edge attribute exists, the weight of the edge is assumed to
be one.

If this is a function, the weight of an edge is the value
returned by the function. The function must accept exactly three
positional arguments: the two endpoints of an edge and the
dictionary of edge attributes for that edge. The function must
return a number or None to indicate a hidden edge.
  - G: <class 'networkx.classes.graph.Graph'> = ? --?
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.shortest_paths.weighted.multi_source_dijkstra_path_length(cutoff=<cutoff_value>, weight=<weight_value>, G=<G_variable>)
