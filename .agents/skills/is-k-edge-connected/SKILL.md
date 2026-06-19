---
name: is-k-edge-connected
description: Tests to see if a graph is k-edge-connected.
---

**Is k edge connected:**
Tests to see if a graph is k-edge-connected.

Is it impossible to disconnect the graph by removing fewer than k edges?
If so, then G is k-edge-connected.
parameters:
  - k: <class 'int'> = ? --edge connectivity to test for
  - G: <class 'networkx.classes.graph.Graph'> = ? --An undirected graph.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.connectivity.edge_augmentation.is_k_edge_connected(k=<k_value>, G=<G_variable>)
