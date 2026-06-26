**Is forest:**
Returns True if `G` is a forest.

A forest is a graph with no undirected cycles.

For directed graphs, `G` is a forest if the underlying graph is a forest.
The underlying graph is obtained by treating each directed edge as a single
undirected edge in a multigraph.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --The graph to test.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.tree.recognition.is_forest(G=<G_variable>)
