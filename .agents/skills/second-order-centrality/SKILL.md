---
name: second-order-centrality
description: Compute the second order centrality for nodes of G.
---

**Second order centrality:**
Compute the second order centrality for nodes of G.

The second order centrality of a given node is the standard deviation of
the return times to that node of a perpetual random walk on G:
parameters:
  - weight: str | None = weight - .
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.algorithms.centrality.second_order.second_order_centrality(weight=<weight_value>, G=<G_variable>)
