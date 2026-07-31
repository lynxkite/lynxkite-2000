**Wiener index:**
Returns the Wiener index of the given graph.

The *Wiener index* of a graph is the sum of the shortest-path
(weighted) distances between each pair of reachable nodes.
For pairs of nodes in undirected graphs, only one orientation
of the pair is counted.
parameters:
  - weight: str | None = ? --If None, every edge has weight 1.
If a string, use this edge attribute as the edge weight.
Any edge attribute not present defaults to 1.
The edge weights are used to computing shortest-path distances.
  - G: <class 'networkx.classes.graph.Graph'> = ? --?
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.wiener.wiener_index(weight=<weight_value>, G=<G_variable>)
