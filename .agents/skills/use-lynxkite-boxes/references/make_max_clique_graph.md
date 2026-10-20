**Make max clique graph:**
Returns the maximal clique graph of the given graph.

The nodes of the maximal clique graph of `G` are the cliques of
`G` and an edge joins two cliques if the cliques are not disjoint.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --?
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.clique.make_max_clique_graph(G=<G_variable>)
