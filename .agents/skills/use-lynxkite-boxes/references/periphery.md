**Periphery:**
Returns the periphery of the graph G.

The periphery is the set of nodes with eccentricity equal to the diameter.
parameters:
  - usebounds: bool | None = ? --If `True`, use extrema bounding (see Notes) when computing the periphery
for undirected graphs. Extrema bounding may accelerate the
distance calculation for some graphs. `usebounds` is ignored if `G` is
directed or if `e` is not `None`. Default is `False`.
  - weight: <class 'str'> = ? --If this is a string, then edge weights will be accessed via the
edge attribute with this key (that is, the weight of the edge
joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
such edge attribute exists, the weight of the edge is assumed to
be one.

If this is a function, the weight of an edge is the value
returned by the function. The function must accept exactly three
positional arguments: the two endpoints of an edge and the
dictionary of edge attributes for that edge. The function must
return a number.

If this is None, every edge has weight/distance/cost 1.

Weights stored as floating point values can lead to small round-off
errors in distances. Use integer weights to avoid this.

Weights should be positive, since they are distances.
  - G: <class 'networkx.classes.graph.Graph'> = ? --A graph
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.distance_measures.periphery(usebounds=<usebounds_value>, weight=<weight_value>, G=<G_variable>)
