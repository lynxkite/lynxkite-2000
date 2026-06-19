---
name: connected-dominating-set
description: Returns a connected dominating set.
---

**Connected dominating set:**
Returns a connected dominating set.

A *dominating set* for a graph *G* with node set *V* is a subset *D* of *V*
such that every node not in *D* is adjacent to at least one member of *D*
[1]_. A *connected dominating set* is a dominating set *C* that induces a
connected subgraph of *G* [2]_.
Note that connected dominating sets are not unique in general and that there
may be other connected dominating sets.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --Undirected connected graph.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.dominating.connected_dominating_set(G=<G_variable>)
