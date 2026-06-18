---
name: networkx-algorithms-centrality-subgraph-alg
description: Collection of operations - Subgraph centrality exp, Subgraph centrality, Communicability betweenness centrality, Estrada index
---

**Subgraph centrality exp:**
Returns the subgraph centrality for each node of G.

Subgraph centrality  of a node `n` is the sum of weighted closed
walks of all lengths starting and ending at node `n`. The weights
decrease with path length. Each closed walk is associated with a
connected subgraph ([1]_).
parameters:
  - normalized: <class 'bool'> = None -
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.centrality.subgraph_alg.subgraph_centrality_exp(normalized=<normalized_value>, G=<G_variable>)

**Subgraph centrality:**
Returns subgraph centrality for each node in G.

Subgraph centrality  of a node `n` is the sum of weighted closed
walks of all lengths starting and ending at node `n`. The weights
decrease with path length. Each closed walk is associated with a
connected subgraph ([1]_).
parameters:
  - normalized: <class 'bool'> = None -
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.centrality.subgraph_alg.subgraph_centrality(normalized=<normalized_value>, G=<G_variable>)

**Communicability betweenness centrality:**
Returns subgraph communicability for all pairs of nodes in G.

Communicability betweenness measure makes use of the number of walks
connecting every pair of nodes as the basis of a betweenness centrality
measure.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.centrality.subgraph_alg.communicability_betweenness_centrality(G=<G_variable>)

**Estrada index:**
Returns the Estrada index of a the graph G.

The Estrada Index is a topological index of folding or 3D "compactness" ([1]_).
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.centrality.subgraph_alg.estrada_index(G=<G_variable>)
