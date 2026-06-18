---
name: local-bridges
description: Iterate over local bridges of `G` optionally computing the span
---

**Local bridges:**
Iterate over local bridges of `G` optionally computing the span

A *local bridge* is an edge whose endpoints have no common neighbors.
That is, the edge is not part of a triangle in the graph.

The *span* of a *local bridge* is the shortest path length between
the endpoints if the local bridge is removed.
parameters:
  - with_span: <class 'bool'> = None -
  - weight: <class 'str'> = None -
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.bridges.local_bridges(with_span=<with_span_value>, weight=<weight_value>, G=<G_variable>)
