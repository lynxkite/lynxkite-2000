---
name: networkx-algorithms-shortest-paths-dense
description: Collection of operations - Floyd–Warshall, Floyd–Warshall predecessor and distance, Floyd–Warshall NumPy
---

**Floyd–Warshall:**
Find all-pairs shortest path lengths using Floyd's algorithm.
parameters:
  - weight: str | None = weight - .
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.algorithms.shortest_paths.dense.floyd_warshall(weight=<weight_value>, G=<G_variable>)

**Floyd–Warshall predecessor and distance:**
Find all-pairs shortest path lengths using Floyd's algorithm.
parameters:
  - weight: str | None = weight - .
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.algorithms.shortest_paths.dense.floyd_warshall_predecessor_and_distance(weight=<weight_value>, G=<G_variable>)

**Floyd–Warshall NumPy:**
Find all-pairs shortest path lengths using Floyd's algorithm.

This algorithm for finding shortest paths takes advantage of
matrix representations of a graph and works well for dense
graphs where all-pairs shortest path lengths are desired.
The results are returned as a NumPy array, distance[i, j],
where i and j are the indexes of two nodes in nodelist.
The entry distance[i, j] is the distance along a shortest
path from i to j. If no path exists the distance is Inf.
parameters:
  - weight: str | None = weight - .
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.algorithms.shortest_paths.dense.floyd_warshall_numpy(weight=<weight_value>, G=<G_variable>)
