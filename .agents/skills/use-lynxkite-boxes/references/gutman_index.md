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
  - weight: str | None = ? --If None, every edge has weight 1.
If a string, use this edge attribute as the edge weight.
Any edge attribute not present defaults to 1.
The edge weights are used to computing shortest-path distances.
  - G: <class 'networkx.classes.graph.Graph'> = ? --?

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.wiener.gutman_index(weight=<weight_value>, G=<G_variable>)
