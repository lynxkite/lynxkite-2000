---
name: networkx-algorithms-shortest-paths-unweighted
description: Collection of operations - Bidirectional shortest path, Single source shortest path, Single target shortest path, All pairs shortest path, All pairs shortest path length, Predecessor
---

**Bidirectional shortest path:**
Returns a list of nodes in a shortest path between source and target.
parameters:
  - source: <class 'str'> = None - .
  - target: <class 'str'> = None - .
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.algorithms.shortest_paths.unweighted.bidirectional_shortest_path(source=<source_value>, target=<target_value>, G=<G_variable>)

**Single source shortest path:**
Compute shortest path between source
and all other nodes reachable from source.
parameters:
  - source: <class 'str'> = None - .
  - cutoff: int | None = None - .
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.algorithms.shortest_paths.unweighted.single_source_shortest_path(source=<source_value>, cutoff=<cutoff_value>, G=<G_variable>)

**Single target shortest path:**
Compute shortest path to target from all nodes that reach target.
parameters:
  - target: <class 'str'> = None - .
  - cutoff: int | None = None - .
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.algorithms.shortest_paths.unweighted.single_target_shortest_path(target=<target_value>, cutoff=<cutoff_value>, G=<G_variable>)

**All pairs shortest path:**
Compute shortest paths between all nodes.
parameters:
  - cutoff: int | None = None - .
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.algorithms.shortest_paths.unweighted.all_pairs_shortest_path(cutoff=<cutoff_value>, G=<G_variable>)

**All pairs shortest path length:**
Computes the shortest path lengths between all nodes in `G`.
parameters:
  - cutoff: int | None = None - .
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.algorithms.shortest_paths.unweighted.all_pairs_shortest_path_length(cutoff=<cutoff_value>, G=<G_variable>)

**Predecessor:**
Returns dict of predecessors for the path from source to all nodes in G.
parameters:
  - source: <class 'str'> = None - .
  - target: str | None = None - .
  - cutoff: int | None = None - .
  - return_seen: bool | None = None - .
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.algorithms.shortest_paths.unweighted.predecessor(source=<source_value>, target=<target_value>, cutoff=<cutoff_value>, return_seen=<return_seen_value>, G=<G_variable>)
