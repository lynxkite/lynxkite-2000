**Eulerize:**
Transforms a graph into an Eulerian graph.

If `G` is Eulerian the result is `G` as a MultiGraph, otherwise the result is a smallest
(in terms of the number of edges) multigraph whose underlying simple graph is `G`.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --An undirected graph
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.euler.eulerize(G=<G_variable>)
