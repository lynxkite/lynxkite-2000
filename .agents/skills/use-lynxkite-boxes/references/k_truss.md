**k-truss:**
Returns the k-truss of `G`.

The k-truss is the maximal induced subgraph of `G` which contains at least
three vertices where every edge is incident to at least `k-2` triangles.
parameters:
  - k: <class 'int'> = ? --The order of the truss
  - G: <class 'networkx.classes.graph.Graph'> = ? --An undirected graph
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.core.k_truss(k=<k_value>, G=<G_variable>)
