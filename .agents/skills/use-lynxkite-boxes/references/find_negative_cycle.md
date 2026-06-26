**Find negative cycle:**
Returns a cycle with negative total weight if it exists.

Bellman-Ford is used to find shortest_paths. That algorithm
stops if there exists a negative cycle. This algorithm
picks up from there and returns the found negative cycle.

The cycle consists of a list of nodes in the cycle order. The last
node equals the first to make it a cycle.
You can look up the edge weights in the original graph. In the case
of multigraphs the relevant edge is the minimal weight edge between
the nodes in the 2-tuple.

If the graph has no negative cycle, a NetworkXError is raised.
parameters:
  - source: <class 'str'> = ? --The search for the negative cycle will start from this node.
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
output_variable = networkx.algorithms.shortest_paths.weighted.find_negative_cycle(source=<source_value>, weight=<weight_value>, G=<G_variable>)
