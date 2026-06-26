**All pairs shortest path:**
Compute shortest paths between all nodes.
parameters:
  - cutoff: int | None = ? --Depth at which to stop the search. Only paths containing at most
``cutoff + 1`` nodes are returned.
  - G: <class 'networkx.classes.graph.Graph'> = ? --?

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.shortest_paths.unweighted.all_pairs_shortest_path(cutoff=<cutoff_value>, G=<G_variable>)
