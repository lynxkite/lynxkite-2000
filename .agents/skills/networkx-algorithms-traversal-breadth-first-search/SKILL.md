---
name: networkx-algorithms-traversal-breadth-first-search
description: Collection of operations - Descendants at distance, BFS layers, BFS labeled edges
---

**Descendants at distance:**
Returns all nodes at a fixed `distance` from `source` in `G`.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --A graph

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.traversal.breadth_first_search.descendants_at_distance(G=<G_variable>)

**BFS layers:**
Returns an iterator of all the layers in breadth-first search traversal.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --A graph over which to find the layers using breadth-first search.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.traversal.breadth_first_search.bfs_layers(G=<G_variable>)

**BFS labeled edges:**
Iterate over edges in a breadth-first search (BFS) labeled by type.

We generate triple of the form (*u*, *v*, *d*), where (*u*, *v*) is the
edge being explored in the breadth-first search and *d* is one of the
strings 'tree', 'forward', 'level', or 'reverse'.  A 'tree' edge is one in
which *v* is first discovered and placed into the layer below *u*.  A
'forward' edge is one in which *u* is on the layer above *v* and *v* has
already been discovered.  A 'level' edge is one in which both *u* and *v*
occur on the same layer.  A 'reverse' edge is one in which *u* is on a layer
below *v*.

We emit each edge exactly once.  In an undirected graph, 'reverse' edges do
not occur, because each is discovered either as a 'tree' or 'forward' edge.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --A graph over which to find the layers using breadth-first search.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.traversal.breadth_first_search.bfs_labeled_edges(G=<G_variable>)
