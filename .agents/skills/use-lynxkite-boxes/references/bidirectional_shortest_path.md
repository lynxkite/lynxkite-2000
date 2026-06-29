**Bidirectional shortest path:**
Returns a list of nodes in a shortest path between source and target.
parameters:
  - source: <class 'str'> = ? --starting node for path
  - target: <class 'str'> = ? --ending node for path
  - G: <class 'networkx.classes.graph.Graph'> = ? --?

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.shortest_paths.unweighted.bidirectional_shortest_path(source=<source_value>, target=<target_value>, G=<G_variable>)
