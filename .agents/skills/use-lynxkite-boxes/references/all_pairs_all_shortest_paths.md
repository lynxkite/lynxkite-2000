**All pairs all shortest paths:**
Compute all shortest paths between all nodes.
parameters:
  - weight: str | None = ? --If None, every edge has weight/distance/cost 1.
If a string, use this edge attribute as the edge weight.
Any edge attribute not present defaults to 1.
If this is a function, the weight of an edge is the value
returned by the function. The function must accept exactly
three positional arguments: the two endpoints of an edge and
the dictionary of edge attributes for that edge.
The function must return a number.
  - method: str | None = dijkstra --The algorithm to use to compute the path lengths.
Supported options: 'dijkstra', 'bellman-ford'.
Other inputs produce a ValueError.
If `weight` is None, unweighted graph methods are used, and this
suggestion is ignored.
  - G: <class 'networkx.classes.graph.Graph'> = ? --?

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.shortest_paths.generic.all_pairs_all_shortest_paths(weight=<weight_value>, method=<method_value>, G=<G_variable>)
