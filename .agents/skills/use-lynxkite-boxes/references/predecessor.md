**Predecessor:**
Returns dict of predecessors for the path from source to all nodes in G.
parameters:
  - source: <class 'str'> = ? --Starting node for path
  - target: str | None = ? --Ending node for path. If provided only predecessors between
source and target are returned
  - cutoff: int | None = ? --Depth to stop the search. Only paths of length <= `cutoff` are
returned.
  - return_seen: bool | None = ? --Whether to return a dictionary, keyed by node, of the level (number of
hops) to reach the node (as seen during breadth-first-search).
  - G: <class 'networkx.classes.graph.Graph'> = ? --?
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.shortest_paths.unweighted.predecessor(source=<source_value>, target=<target_value>, cutoff=<cutoff_value>, return_seen=<return_seen_value>, G=<G_variable>)
