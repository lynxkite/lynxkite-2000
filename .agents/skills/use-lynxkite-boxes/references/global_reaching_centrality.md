**Global reaching centrality:**
Returns the global reaching centrality of a directed graph.

The *global reaching centrality* of a weighted directed graph is the
average over all nodes of the difference between the local reaching
centrality of the node and the greatest local reaching centrality of
any node in the graph [1]_. For more information on the local
reaching centrality, see :func:`local_reaching_centrality`.
Informally, the local reaching centrality is the proportion of the
graph that is reachable from the neighbors of the node.
parameters:
  - weight: str | None = ? --Attribute to use for edge weights. If ``None``, each edge weight
is assumed to be one. A higher weight implies a stronger
connection between nodes and a *shorter* path length.
  - normalized: bool | None = ? --Whether to normalize the edge weights by the total sum of edge
weights.
  - G: <class 'networkx.classes.graph.Graph'> = ? --A networkx DiGraph.
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.centrality.reaching.global_reaching_centrality(weight=<weight_value>, normalized=<normalized_value>, G=<G_variable>)
