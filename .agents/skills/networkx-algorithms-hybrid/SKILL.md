---
name: networkx-algorithms-hybrid
description: Collection of operations - KL connected subgraph, Is KL connected
---

**KL connected subgraph:**
Returns the maximum locally `(k, l)`-connected subgraph of `G`.

A graph is locally `(k, l)`-connected if for each edge `(u, v)` in the
graph there are at least `l` edge-disjoint paths of length at most `k`
joining `u` to `v`.
parameters:
  - k: <class 'int'> = None -
  - l: <class 'int'> = None -
  - low_memory: <class 'bool'> = None -
  - same_as_graph: <class 'bool'> = None -
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.hybrid.kl_connected_subgraph(k=<k_value>, l=<l_value>, low_memory=<low_memory_value>, same_as_graph=<same_as_graph_value>, G=<G_variable>)

**Is KL connected:**
Returns True if and only if `G` is locally `(k, l)`-connected.

A graph is locally `(k, l)`-connected if for each edge `(u, v)` in the
graph there are at least `l` edge-disjoint paths of length at most `k`
joining `u` to `v`.
parameters:
  - k: <class 'int'> = None -
  - l: <class 'int'> = None -
  - low_memory: <class 'bool'> = None -
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.hybrid.is_kl_connected(k=<k_value>, l=<l_value>, low_memory=<low_memory_value>, G=<G_variable>)
