**In degree centrality:**
Compute the in-degree centrality for nodes.

The in-degree centrality for a node v is the fraction of nodes its
incoming edges are connected to.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --A NetworkX graph

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.centrality.degree_alg.in_degree_centrality(G=<G_variable>)
