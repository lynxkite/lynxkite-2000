**Inverse line graph:**
Returns the inverse line graph of graph G.

If H is a graph, and G is the line graph of H, such that G = L(H).
Then H is the inverse line graph of G.

Not all graphs are line graphs and these do not have an inverse line graph.
In these cases this function raises a NetworkXError.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --A NetworkX Graph
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.generators.line.inverse_line_graph(G=<G_variable>)
