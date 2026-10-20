**Floyd–Warshall:**
Find all-pairs shortest path lengths using Floyd's algorithm.
parameters:
  - weight: str | None = weight --Edge data key corresponding to the edge weight.
  - G: <class 'networkx.classes.graph.Graph'> = ? --?
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.shortest_paths.dense.floyd_warshall(weight=<weight_value>, G=<G_variable>)
