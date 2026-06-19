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
  - copy: bool | None = ? --If this is True, then this function returns a new graph with
the stochastic reweighting. Otherwise, the original graph is
modified in-place (and also returned, for convenience).
  - weight: <class 'str'> = weight --Edge attribute key used for reading the existing weight and
setting the new weight.  If no attribute with this key is found
for an edge, then the edge weight is assumed to be 1. If an edge
has a weight, it must be a positive number.
  - G: <class 'networkx.classes.graph.Graph'> = ? --A :class:`~networkx.DiGraph` or :class:`~networkx.MultiDiGraph`.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.stochastic.stochastic_graph(copy=<copy_value>, weight=<weight_value>, G=<G_variable>)
