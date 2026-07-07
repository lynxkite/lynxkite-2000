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
  - weight: str | None = ? --If None, every edge has weight 1.
If a string, use this edge attribute as the edge weight.
Any edge attribute not present defaults to 1.
The edge weights are used to computing shortest-path distances.
  - G: <class 'networkx.classes.graph.Graph'> = ? --The undirected graph of interest.
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.wiener.schultz_index(weight=<weight_value>, G=<G_variable>)
