---
name: networkx-algorithms-communicability-alg
description: Collection of operations - Communicability, Communicability exp
---

**Communicability:**
Returns communicability between all pairs of nodes in G.

The communicability between pairs of nodes in G is the sum of
walks of different lengths starting at node u and ending at node v.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.communicability_alg.communicability(G=<G_variable>)

**Communicability exp:**
Returns communicability between all pairs of nodes in G.

Communicability between pair of node (u,v) of node in G is the sum of
walks of different lengths starting at node u and ending at node v.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.communicability_alg.communicability_exp(G=<G_variable>)
