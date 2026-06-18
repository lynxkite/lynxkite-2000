---
name: networkx-algorithms-connectivity-edge-kcomponents
description: Collection of operations - K edge components, K edge subgraphs
---

**K edge components:**
Generates nodes in each maximal k-edge-connected component in G.
parameters:
  - k: <class 'int'> = None - .
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.algorithms.connectivity.edge_kcomponents.k_edge_components(k=<k_value>, G=<G_variable>)

**K edge subgraphs:**
Generates nodes in each maximal k-edge-connected subgraph in G.
parameters:
  - k: <class 'int'> = None - .
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.algorithms.connectivity.edge_kcomponents.k_edge_subgraphs(k=<k_value>, G=<G_variable>)
