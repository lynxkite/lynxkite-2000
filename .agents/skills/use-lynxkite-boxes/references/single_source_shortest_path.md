**Single source shortest path:**
Compute shortest path between source
and all other nodes reachable from source.
parameters:
  - source: <class 'str'> = ? --Starting node for path
  - cutoff: int | None = ? --Depth to stop the search. Only target nodes where the shortest path to
this node from the source node contains <= ``cutoff + 1`` nodes will be
included in the returned results.
  - G: <class 'networkx.classes.graph.Graph'> = ? --?
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.shortest_paths.unweighted.single_source_shortest_path(source=<source_value>, cutoff=<cutoff_value>, G=<G_variable>)
