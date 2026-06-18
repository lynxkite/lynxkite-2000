---
name: complete-bipartite-graph
description: Returns the complete bipartite graph `K_{n_1,n_2}`.
---

**Complete bipartite graph:**
Returns the complete bipartite graph `K_{n_1,n_2}`.

The graph is composed of two partitions with nodes 0 to (n1 - 1)
in the first and nodes n1 to (n1 + n2 - 1) in the second.
Each node in the first is connected to each node in the second.
parameters:


usage:
output_variable = networkx.algorithms.bipartite.generators.complete_bipartite_graph()
