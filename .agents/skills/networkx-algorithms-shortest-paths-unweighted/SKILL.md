---
name: networkx-algorithms-shortest-paths-unweighted
description: Collection of operations - Bidirectional shortest path, Single source shortest path, Single target shortest path, All pairs shortest path, All pairs shortest path length, Predecessor
---

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
