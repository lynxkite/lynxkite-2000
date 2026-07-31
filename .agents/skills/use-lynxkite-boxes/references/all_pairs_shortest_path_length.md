**All pairs shortest path length:**
Computes the shortest path lengths between all nodes in `G`.
parameters:
  - cutoff: int | None = ? --Depth at which to stop the search. Only paths of length at most
`cutoff` (i.e. paths containing <= ``cutoff + 1`` nodes) are returned.
  - G: <class 'networkx.classes.graph.Graph'> = ? --?
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.shortest_paths.unweighted.all_pairs_shortest_path_length(cutoff=<cutoff_value>, G=<G_variable>)
