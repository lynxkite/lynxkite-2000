**Maximal matching:**
Find a maximal matching in the graph.

A matching is a subset of edges in which no node occurs more than once.
A maximal matching cannot add more edges and still be a matching.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --Undirected graph
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.matching.maximal_matching(G=<G_variable>)
