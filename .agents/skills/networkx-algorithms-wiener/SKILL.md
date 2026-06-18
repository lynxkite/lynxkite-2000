---
name: networkx-algorithms-wiener
description: Collection of operations - Wiener index, Schultz index, Gutman index, Hyper wiener index
---

**Wiener index:**
Returns the Wiener index of the given graph.

The *Wiener index* of a graph is the sum of the shortest-path
(weighted) distances between each pair of reachable nodes.
For pairs of nodes in undirected graphs, only one orientation
of the pair is counted.
parameters:
  - weight: str | None = None -
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.wiener.wiener_index(weight=<weight_value>, G=<G_variable>)

**Schultz index:**
Returns the Schultz Index (of the first kind) of `G`

The *Schultz Index* [3]_ of a graph is the sum over all node pairs of
distances times the sum of degrees. Consider an undirected graph `G`.
For each node pair ``(u, v)`` compute ``dist(u, v) * (deg(u) + deg(v)``
where ``dist`` is the shortest path length between two nodes and ``deg``
is the degree of a node.

The Schultz Index is the sum of these quantities over all (unordered)
pairs of nodes.
parameters:
  - weight: str | None = None -
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.wiener.schultz_index(weight=<weight_value>, G=<G_variable>)

**Gutman index:**
Returns the Gutman Index for the graph `G`.

The *Gutman Index* measures the topology of networks, especially for molecule
networks of atoms connected by bonds [1]_. It is also called the Schultz Index
of the second kind [2]_.

Consider an undirected graph `G` with node set ``V``.
The Gutman Index of a graph is the sum over all (unordered) pairs of nodes
of nodes ``(u, v)``, with distance ``dist(u, v)`` and degrees ``deg(u)``
and ``deg(v)``, of ``dist(u, v) * deg(u) * deg(v)``
parameters:
  - weight: str | None = None -
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.wiener.gutman_index(weight=<weight_value>, G=<G_variable>)

**Hyper wiener index:**
Returns the Hyper-Wiener index of the graph `G`.

The Hyper-Wiener index of a connected graph `G` is defined as

.. math::
    WW(G) = \frac{1}{2} \sum_{u,v \in V(G)} (d(u,v) + d(u,v)^2)

where ``d(u, v)`` is the shortest-path distance between nodes ``u`` and ``v``.
parameters:
  - weight: str | None = None -
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.wiener.hyper_wiener_index(weight=<weight_value>, G=<G_variable>)
