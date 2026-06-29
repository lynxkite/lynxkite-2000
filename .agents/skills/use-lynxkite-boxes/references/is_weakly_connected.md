**Is weakly connected:**
Test directed graph for weak connectivity.

A directed graph is weakly connected if and only if the graph
is connected when the direction of the edge between nodes is ignored.

Note that if a graph is strongly connected (i.e. the graph is connected
even when we account for directionality), it is by definition weakly
connected as well.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --A directed graph.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.components.weakly_connected.is_weakly_connected(G=<G_variable>)
