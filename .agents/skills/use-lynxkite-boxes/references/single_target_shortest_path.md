**Single target shortest path:**
Compute shortest path to target from all nodes that reach target.
parameters:
  - target: <class 'str'> = ? --Target node for path
  - cutoff: int | None = ? --Depth to stop the search. Only source nodes where the shortest path
from this node to the target node contains <= ``cutoff + 1`` nodes will
be included in the returned results.
  - G: <class 'networkx.classes.graph.Graph'> = ? --?
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.shortest_paths.unweighted.single_target_shortest_path(target=<target_value>, cutoff=<cutoff_value>, G=<G_variable>)
