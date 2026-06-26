**k-corona:**
Returns the k-corona of G.

The k-corona is the subgraph of nodes in the k-core which have
exactly k neighbors in the k-core.
parameters:
  - k: <class 'int'> = ? --The order of the corona.
  - G: <class 'networkx.classes.graph.Graph'> = ? --A graph or directed graph

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.core.k_corona(k=<k_value>, G=<G_variable>)
