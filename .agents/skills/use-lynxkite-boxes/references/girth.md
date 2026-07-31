**Girth:**
Returns the girth of the graph.

The girth of a graph is the length of its shortest cycle, or infinity if
the graph is acyclic. The algorithm follows the description given on the
Wikipedia page [1]_, and runs in time O(mn) on a graph with m edges and n
nodes.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --?
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.cycles.girth(G=<G_variable>)
