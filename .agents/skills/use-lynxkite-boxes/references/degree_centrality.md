**Degree centrality:**
Compute the degree centrality for nodes.

The degree centrality for a node v is the fraction of nodes it
is connected to.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --A networkx graph
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.centrality.degree_alg.degree_centrality(G=<G_variable>)
