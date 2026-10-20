**Is strongly regular:**
Returns True if and only if the given graph is strongly
regular.

An undirected graph is *strongly regular* if

* it is regular,
* each pair of adjacent vertices has the same number of neighbors in
  common,
* each pair of nonadjacent vertices has the same number of neighbors
  in common.

Each strongly regular graph is a distance-regular graph.
Conversely, if a distance-regular graph has diameter two, then it is
a strongly regular graph. For more information on distance-regular
graphs, see :func:`is_distance_regular`.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --An undirected graph.
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.distance_regular.is_strongly_regular(G=<G_variable>)
