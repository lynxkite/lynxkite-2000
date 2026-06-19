**Second order centrality:**
Compute the second order centrality for nodes of G.

The second order centrality of a given node is the standard deviation of
the return times to that node of a perpetual random walk on G:
parameters:
  - weight: str | None = weight --The name of an edge attribute that holds the numerical value
used as a weight. If None then each edge has weight 1.
  - G: <class 'networkx.classes.graph.Graph'> = ? --A NetworkX connected and undirected graph.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.centrality.second_order.second_order_centrality(weight=<weight_value>, G=<G_variable>)
