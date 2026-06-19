---
name: networkx-algorithms-broadcasting
description: Collection of operations - Tree broadcast center, Tree broadcast time
---

**Tree broadcast center:**
Return the broadcast center of a tree.

The broadcast center of a graph `G` denotes the set of nodes having
minimum broadcast time [1]_. This function implements a linear algorithm
for determining the broadcast center of a tree with ``n`` nodes. As a
by-product, it also determines the broadcast time from the broadcast center.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --The graph should be an undirected tree.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.broadcasting.tree_broadcast_center(G=<G_variable>)

**Tree broadcast time:**
Return the minimum broadcast time of a (node in a) tree.

The minimum broadcast time of a node is defined as the minimum amount
of time required to complete broadcasting starting from that node.
The broadcast time of a graph is the maximum over
all nodes of the minimum broadcast time from that node [1]_.
This function returns the minimum broadcast time of `node`.
If `node` is `None`, the broadcast time for the graph is returned.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --The graph should be an undirected tree.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.broadcasting.tree_broadcast_time(G=<G_variable>)
