---
name: networkx-algorithms-shortest-paths-generic
description: Collection of operations - Shortest path, All pairs all shortest paths, Shortest path length, Average shortest path length
---

**Shortest path:**
Compute shortest paths in the graph.
parameters:
  - weight: str | None = None -
  - method: str | None = dijkstra -
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.shortest_paths.generic.shortest_path(weight=<weight_value>, method=<method_value>, G=<G_variable>)

**All pairs all shortest paths:**
Compute all shortest paths between all nodes.
parameters:
  - weight: str | None = None -
  - method: str | None = dijkstra -
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.shortest_paths.generic.all_pairs_all_shortest_paths(weight=<weight_value>, method=<method_value>, G=<G_variable>)

**Shortest path length:**
Compute shortest path lengths in the graph.
parameters:
  - weight: str | None = None -
  - method: str | None = dijkstra -
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.shortest_paths.generic.shortest_path_length(weight=<weight_value>, method=<method_value>, G=<G_variable>)

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
  - weight: str | None = None -
  - method: str | None = None -
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.shortest_paths.generic.average_shortest_path_length(weight=<weight_value>, method=<method_value>, G=<G_variable>)
