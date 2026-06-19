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
  - weight: str | None = weight --Edge data key corresponding to the edge weight.
  - G: <class 'networkx.classes.graph.Graph'> = ? --?

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.shortest_paths.dense.floyd_warshall_numpy(weight=<weight_value>, G=<G_variable>)
