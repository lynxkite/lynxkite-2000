**BFS layers:**
Returns an iterator of all the layers in breadth-first search traversal.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --A graph over which to find the layers using breadth-first search.
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.traversal.breadth_first_search.bfs_layers(G=<G_variable>)
