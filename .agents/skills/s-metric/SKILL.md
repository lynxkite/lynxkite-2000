---
name: s-metric
description: Returns the s-metric [1]_ of graph.
---

**s-metric:**
Returns the s-metric [1]_ of graph.

The s-metric is defined as the sum of the products ``deg(u) * deg(v)``
for every edge ``(u, v)`` in `G`.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --The graph used to compute the s-metric.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.smetric.s_metric(G=<G_variable>)
