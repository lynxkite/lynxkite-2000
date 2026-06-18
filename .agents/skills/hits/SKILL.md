---
name: hits
description: Returns HITS hubs and authorities values for nodes.
---

**Hits:**
Returns HITS hubs and authorities values for nodes.

The HITS algorithm computes two numbers for a node.
Authorities estimates the node value based on the incoming links.
Hubs estimates the node value based on outgoing links.
parameters:
  - max_iter: int | None = 100 -
  - tol: float | None = 1e-08 -
  - normalized: <class 'bool'> = None -
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.link_analysis.hits_alg.hits(max_iter=<max_iter_value>, tol=<tol_value>, normalized=<normalized_value>, G=<G_variable>)
