**k-crust:**
Returns the k-crust of G.

The k-crust is the graph G with the edges of the k-core removed
and isolated nodes found after the removal of edges are also removed.
parameters:
  - k: int | None = ? --The order of the shell. If not specified return the main crust.
  - G: <class 'networkx.classes.graph.Graph'> = ? --A graph or directed graph.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.core.k_crust(k=<k_value>, G=<G_variable>)
