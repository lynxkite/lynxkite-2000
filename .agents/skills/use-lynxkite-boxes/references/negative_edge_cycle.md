**Negative edge cycle:**
Returns True if there exists a negative edge cycle anywhere in G.
parameters:
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
  - heuristic: <class 'bool'> = ? --Determines whether to use a heuristic to early detect negative
cycles at a negligible cost. In case of graphs with a negative cycle,
the performance of detection increases by at least an order of magnitude.
  - G: <class 'networkx.classes.graph.Graph'> = ? --?

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.shortest_paths.weighted.negative_edge_cycle(weight=<weight_value>, heuristic=<heuristic_value>, G=<G_variable>)
