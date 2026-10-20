**Bellman–Ford predecessor and distance:**
Compute shortest path lengths and predecessors on shortest paths
in weighted graphs.

The algorithm has a running time of $O(mn)$ where $n$ is the number of
nodes and $m$ is the number of edges.  It is slower than Dijkstra but
can handle negative edge weights.

If a negative cycle is detected, you can use :func:`find_negative_cycle`
to return the cycle and examine it. Shortest paths are not defined when
a negative cycle exists because once reached, the path can cycle forever
to build up arbitrarily low weights.
parameters:
  - source: <class 'str'> = ? --Starting node for path
  - target: str | None = ? --Ending node for path
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
cycles at a hopefully negligible cost.
  - G: <class 'networkx.classes.graph.Graph'> = ? --The algorithm works for all types of graphs, including directed
graphs and multigraphs.
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.shortest_paths.weighted.bellman_ford_predecessor_and_distance(source=<source_value>, target=<target_value>, weight=<weight_value>, heuristic=<heuristic_value>, G=<G_variable>)
