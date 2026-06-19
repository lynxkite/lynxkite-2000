---
name: networkx-algorithms-planarity
description: Collection of operations - Check planarity, Is planar
---

**Check planarity:**
Check if a graph is planar and return a counterexample or an embedding.

A graph is planar iff it can be drawn in a plane without
any edge intersections.
parameters:
  - counterexample: <class 'bool'> = ? --A Kuratowski subgraph (to proof non planarity) is only returned if set
to true.
  - G: <class 'networkx.classes.graph.Graph'> = ? --?

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.planarity.check_planarity(counterexample=<counterexample_value>, G=<G_variable>)

**Is planar:**
Returns True if and only if `G` is planar.

A graph is *planar* iff it can be drawn in a plane without
any edge intersections.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --?

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.planarity.is_planar(G=<G_variable>)
