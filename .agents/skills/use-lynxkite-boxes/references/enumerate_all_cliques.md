**Enumerate all cliques:**
Returns all cliques in an undirected graph.

This function returns an iterator over cliques, each of which is a
list of nodes. The iteration is ordered by cardinality of the
cliques: first all cliques of size one, then all cliques of size
two, etc.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --An undirected graph.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.clique.enumerate_all_cliques(G=<G_variable>)
