---
name: networkx-algorithms-centrality-degree-alg
description: Collection of operations - Degree centrality, In degree centrality, Out degree centrality
---

**Degree centrality:**
Compute the degree centrality for nodes.

The degree centrality for a node v is the fraction of nodes it
is connected to.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.algorithms.centrality.degree_alg.degree_centrality(G=<G_variable>)

**In degree centrality:**
Compute the in-degree centrality for nodes.

The in-degree centrality for a node v is the fraction of nodes its
incoming edges are connected to.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.algorithms.centrality.degree_alg.in_degree_centrality(G=<G_variable>)

**Out degree centrality:**
Compute the out-degree centrality for nodes.

The out-degree centrality for a node v is the fraction of nodes its
outgoing edges are connected to.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.algorithms.centrality.degree_alg.out_degree_centrality(G=<G_variable>)
