**Is biconnected:**
Returns True if the graph is biconnected, False otherwise.

A graph is biconnected if, and only if, it cannot be disconnected by
removing only one node (and all edges incident on that node).  If
removing a node increases the number of disconnected components
in the graph, that node is called an articulation point, or cut
vertex.  A biconnected graph has no articulation points.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --An undirected graph.
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.components.biconnected.is_biconnected(G=<G_variable>)
