**Average shortest path length:**
Returns the average shortest path length.

The average shortest path length is

.. math::

   a =\sum_{\substack{s,t \in V \\ s\neq t}} \frac{d(s, t)}{n(n-1)}

where `V` is the set of nodes in `G`,
`d(s, t)` is the shortest path from `s` to `t`,
and `n` is the number of nodes in `G`.

.. versionchanged:: 3.0
   An exception is raised for directed graphs that are not strongly
   connected.
parameters:
  - weight: str | None = ? --If None, every edge has weight/distance/cost 1.
If a string, use this edge attribute as the edge weight.
Any edge attribute not present defaults to 1.
If this is a function, the weight of an edge is the value
returned by the function. The function must accept exactly
three positional arguments: the two endpoints of an edge and
the dictionary of edge attributes for that edge.
The function must return a number.
  - method: str | None = ? --The algorithm to use to compute the path lengths.
Supported options are 'unweighted', 'dijkstra', 'bellman-ford',
'floyd-warshall' and 'floyd-warshall-numpy'.
Other method values produce a ValueError.
The default method is 'unweighted' if `weight` is None,
otherwise the default method is 'dijkstra'.
  - G: <class 'networkx.classes.graph.Graph'> = ? --?

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.shortest_paths.generic.average_shortest_path_length(weight=<weight_value>, method=<method_value>, G=<G_variable>)
