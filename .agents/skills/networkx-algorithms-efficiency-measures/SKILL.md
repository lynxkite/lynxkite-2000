---
name: networkx-algorithms-efficiency-measures
description: Collection of operations - Local efficiency, Global efficiency
---

**Local efficiency:**
Returns the average local efficiency of the graph.

The *efficiency* of a pair of nodes in a graph is the multiplicative
inverse of the shortest path distance between the nodes. The *local
efficiency* of a node in the graph is the average global efficiency of the
subgraph induced by the neighbors of the node. The *average local
efficiency* is the average of the local efficiencies of each node [1]_.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.efficiency_measures.local_efficiency(G=<G_variable>)

**Global efficiency:**
Returns the average global efficiency of the graph.

The *efficiency* of a pair of nodes in a graph is the multiplicative
inverse of the shortest path distance between the nodes. The *average
global efficiency* of a graph is the average efficiency of all pairs of
nodes [1]_.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.efficiency_measures.global_efficiency(G=<G_variable>)
