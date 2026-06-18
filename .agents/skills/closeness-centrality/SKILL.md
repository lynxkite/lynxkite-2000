---
name: closeness-centrality
description: Compute closeness centrality for nodes.
---

**Closeness centrality:**
Compute closeness centrality for nodes.

Closeness centrality [1]_ of a node `u` is the reciprocal of the
average shortest path distance to `u` over all `n-1` reachable nodes.

.. math::

    C(u) = \frac{n - 1}{\sum_{v=1}^{n-1} d(v, u)},

where `d(v, u)` is the shortest-path distance between `v` and `u`,
and `n-1` is the number of nodes reachable from `u`. Notice that the
closeness distance function computes the incoming distance to `u`
for directed graphs. To use outward distance, act on `G.reverse()`.

Notice that higher values of closeness indicate higher centrality.

Wasserman and Faust propose an improved formula for graphs with
more than one connected component. The result is "a ratio of the
fraction of actors in the group who are reachable, to the average
distance" from the reachable actors [2]_. You might think this
scale factor is inverted but it is not. As is, nodes from small
components receive a smaller closeness value. Letting `N` denote
the number of nodes in the graph,

.. math::

    C_{WF}(u) = \frac{n-1}{N-1} \frac{n - 1}{\sum_{v=1}^{n-1} d(v, u)},
parameters:
  - wf_improved: bool | None = None - .
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.algorithms.centrality.closeness.closeness_centrality(wf_improved=<wf_improved_value>, G=<G_variable>)
