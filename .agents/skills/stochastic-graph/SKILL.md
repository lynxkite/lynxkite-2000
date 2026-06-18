---
name: stochastic-graph
description: Returns a right-stochastic representation of directed graph `G`.
---

**Stochastic graph:**
Returns a right-stochastic representation of directed graph `G`.

A right-stochastic graph is a weighted digraph in which for each
node, the sum of the weights of all the out-edges of that node is
1. If the graph is already weighted (for example, via a 'weight'
edge attribute), the reweighting takes that into account.
parameters:
  - copy: bool | None = None -
  - weight: <class 'str'> = weight -
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.generators.stochastic.stochastic_graph(copy=<copy_value>, weight=<weight_value>, G=<G_variable>)
