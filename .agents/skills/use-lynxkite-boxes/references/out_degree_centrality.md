**Out degree centrality:**
Compute the out-degree centrality for nodes.

The out-degree centrality for a node v is the fraction of nodes its
outgoing edges are connected to.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --A NetworkX graph
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.centrality.degree_alg.out_degree_centrality(G=<G_variable>)
