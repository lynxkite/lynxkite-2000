**Is tree:**
Returns True if `G` is a tree.

A tree is a connected graph with no undirected cycles.

For directed graphs, `G` is a tree if the underlying graph is a tree. The
underlying graph is obtained by treating each directed edge as a single
undirected edge in a multigraph.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --The graph to test.
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.tree.recognition.is_tree(G=<G_variable>)
