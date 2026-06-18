---
name: edge-bfs
description: A directed, breadth-first-search of edges in `G`, beginning at `source`.
---

**Edge BFS:**
A directed, breadth-first-search of edges in `G`, beginning at `source`.

Yield the edges of G in a breadth-first-search order continuing until
all edges are generated.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.algorithms.traversal.edgebfs.edge_bfs(G=<G_variable>)
