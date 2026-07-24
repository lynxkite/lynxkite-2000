**Condensation:**
Returns the condensation of G.

The condensation of G is the graph with each of the strongly connected
components contracted into a single node.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --A directed graph.
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.components.strongly_connected.condensation(G=<G_variable>)
