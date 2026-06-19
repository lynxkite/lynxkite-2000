---
name: networkx-algorithms-shortest-paths-weighted
description: Collection of operations - Dijkstra path length, Single source Dijkstra, Single source Dijkstra path length, Multi source Dijkstra, Multi source Dijkstra path, Multi source Dijkstra path length, All pairs Dijkstra, All pairs Dijkstra path, All pairs Dijkstra path length, Dijkstra predecessor and distance, Bellman–Ford path length, Single source Bellman–Ford, Single source Bellman–Ford path length, All pairs Bellman–Ford path, All pairs Bellman–Ford path length, Bellman–Ford predecessor and distance, Negative edge cycle, Find negative cycle, Goldberg Radzik, Johnson
---

**Dijkstra path length:**
Returns the shortest weighted path length in G from source to target.

Uses Dijkstra's Method to compute the shortest weighted path length
between two nodes in a graph.
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
return a number or None to indicate a hidden edge.
  - G: <class 'networkx.classes.graph.Graph'> = ? --?

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.shortest_paths.weighted.dijkstra_path_length(source=<source_value>, target=<target_value>, weight=<weight_value>, G=<G_variable>)

**Single source Dijkstra:**
Find shortest weighted paths and lengths from a source node.

Compute the shortest path length between source and all other
reachable nodes for a weighted graph.

Uses Dijkstra's algorithm to compute shortest paths and lengths
between a source and all other reachable nodes in a weighted graph.
parameters:
  - source: <class 'str'> = ? --Starting node for path
  - target: str | None = ? --Ending node for path
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
output_variable = networkx.algorithms.shortest_paths.weighted.single_source_dijkstra(source=<source_value>, target=<target_value>, cutoff=<cutoff_value>, weight=<weight_value>, G=<G_variable>)

**Single source Dijkstra path length:**
Find shortest weighted path lengths in G from a source node.

Compute the shortest path length between source and all other
reachable nodes for a weighted graph.
parameters:
  - source: <class 'str'> = ? --Starting node for path
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
output_variable = networkx.algorithms.shortest_paths.weighted.single_source_dijkstra_path_length(source=<source_value>, cutoff=<cutoff_value>, weight=<weight_value>, G=<G_variable>)

**Multi source Dijkstra:**
Find shortest weighted paths and lengths from a given set of
source nodes.

Uses Dijkstra's algorithm to compute the shortest paths and lengths
between one of the source nodes and the given `target`, or all other
reachable nodes if not specified, for a weighted graph.
parameters:
  - target: str | None = ? --Ending node for path
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
output_variable = networkx.algorithms.shortest_paths.weighted.multi_source_dijkstra(target=<target_value>, cutoff=<cutoff_value>, weight=<weight_value>, G=<G_variable>)

**Multi source Dijkstra path:**
Find shortest weighted paths in G from a given set of source
nodes.

Compute shortest path between any of the source nodes and all other
reachable nodes for a weighted graph.
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
output_variable = networkx.algorithms.shortest_paths.weighted.multi_source_dijkstra_path(cutoff=<cutoff_value>, weight=<weight_value>, G=<G_variable>)

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

**All pairs Dijkstra:**
Find shortest weighted paths and lengths between all nodes.
parameters:
  - cutoff: float | None = ? --Length (sum of edge weights) at which the search is stopped.
If cutoff is provided, only return paths with summed weight <= cutoff.
  - weight: <class 'str'> = weight --If this is a string, then edge weights will be accessed via the
edge attribute with this key (that is, the weight of the edge
joining `u` to `v` will be ``G.edge[u][v][weight]``). If no
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
output_variable = networkx.algorithms.shortest_paths.weighted.all_pairs_dijkstra(cutoff=<cutoff_value>, weight=<weight_value>, G=<G_variable>)

**All pairs Dijkstra path:**
Compute shortest paths between all nodes in a weighted graph.
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
output_variable = networkx.algorithms.shortest_paths.weighted.all_pairs_dijkstra_path(cutoff=<cutoff_value>, weight=<weight_value>, G=<G_variable>)

**All pairs Dijkstra path length:**
Compute shortest path lengths between all nodes in a weighted graph.
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
output_variable = networkx.algorithms.shortest_paths.weighted.all_pairs_dijkstra_path_length(cutoff=<cutoff_value>, weight=<weight_value>, G=<G_variable>)

**Dijkstra predecessor and distance:**
Compute weighted shortest path length and predecessors.

Uses Dijkstra's Method to obtain the shortest weighted paths
and return dictionaries of predecessors for each node and
distance for each node from the `source`.
parameters:
  - source: <class 'str'> = ? --Starting node for path
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
output_variable = networkx.algorithms.shortest_paths.weighted.dijkstra_predecessor_and_distance(source=<source_value>, cutoff=<cutoff_value>, weight=<weight_value>, G=<G_variable>)

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

**Single source Bellman–Ford:**
Compute shortest paths and lengths in a weighted graph G.

Uses Bellman-Ford algorithm for shortest paths.
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
  - G: <class 'networkx.classes.graph.Graph'> = ? --?

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.shortest_paths.weighted.single_source_bellman_ford(source=<source_value>, target=<target_value>, weight=<weight_value>, G=<G_variable>)

**Single source Bellman–Ford path length:**
Compute the shortest path length between source and all other
reachable nodes for a weighted graph.
parameters:
  - source: <class 'str'> = ? --Starting node for path
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
output_variable = networkx.algorithms.shortest_paths.weighted.single_source_bellman_ford_path_length(source=<source_value>, weight=<weight_value>, G=<G_variable>)

**All pairs Bellman–Ford path:**
Compute shortest paths between all nodes in a weighted graph.
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
  - G: <class 'networkx.classes.graph.Graph'> = ? --?

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.shortest_paths.weighted.all_pairs_bellman_ford_path(weight=<weight_value>, G=<G_variable>)

**All pairs Bellman–Ford path length:**
Compute shortest path lengths between all nodes in a weighted graph.
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
  - G: <class 'networkx.classes.graph.Graph'> = ? --?

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.shortest_paths.weighted.all_pairs_bellman_ford_path_length(weight=<weight_value>, G=<G_variable>)

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

**Goldberg Radzik:**
Compute shortest path lengths and predecessors on shortest paths
in weighted graphs.

The algorithm has a running time of $O(mn)$ where $n$ is the number of
nodes and $m$ is the number of edges.  It is slower than Dijkstra but
can handle negative edge weights.
parameters:
  - source: <class 'str'> = ? --Starting node for path
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
  - G: <class 'networkx.classes.graph.Graph'> = ? --The algorithm works for all types of graphs, including directed
graphs and multigraphs.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.shortest_paths.weighted.goldberg_radzik(source=<source_value>, weight=<weight_value>, G=<G_variable>)

**Johnson:**
Uses Johnson's Algorithm to compute shortest paths.

Johnson's Algorithm finds a shortest path between each pair of
nodes in a weighted graph even if negative weights are present.
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
  - G: <class 'networkx.classes.graph.Graph'> = ? --?

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.shortest_paths.weighted.johnson(weight=<weight_value>, G=<G_variable>)
