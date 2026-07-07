**All triangles:**
Yields all unique triangles in an undirected graph.

A triangle is a set of three distinct nodes where each node is connected to
the other two.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --An undirected graph.
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.cluster.all_triangles(G=<G_variable>)
