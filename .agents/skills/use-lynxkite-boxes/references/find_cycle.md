**Find cycle:**
Returns a cycle found via depth-first traversal.

The cycle is a list of edges indicating the cyclic path.
Orientation of directed edges is controlled by `orientation`.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --A directed/undirected graph/multigraph.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.cycles.find_cycle(G=<G_variable>)
