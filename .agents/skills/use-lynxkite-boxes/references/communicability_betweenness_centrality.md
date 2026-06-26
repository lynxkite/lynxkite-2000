**Communicability betweenness centrality:**
Returns subgraph communicability for all pairs of nodes in G.

Communicability betweenness measure makes use of the number of walks
connecting every pair of nodes as the basis of a betweenness centrality
measure.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --?

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.centrality.subgraph_alg.communicability_betweenness_centrality(G=<G_variable>)
