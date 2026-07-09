**Is strongly connected:**
Test directed graph for strong connectivity.

A directed graph is strongly connected if and only if every vertex in
the graph is reachable from every other vertex.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --A directed graph.
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.components.strongly_connected.is_strongly_connected(G=<G_variable>)
