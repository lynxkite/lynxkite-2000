---
name: is-k-edge-connected
description: Tests to see if a graph is k-edge-connected.
---

**Is k edge connected:**
Tests to see if a graph is k-edge-connected.

Is it impossible to disconnect the graph by removing fewer than k edges?
If so, then G is k-edge-connected.
parameters:
  - k: <class 'int'> = None -
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.connectivity.edge_augmentation.is_k_edge_connected(k=<k_value>, G=<G_variable>)
