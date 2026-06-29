**Edge DFS:**
A directed, depth-first-search of edges in `G`, beginning at `source`.

Yield the edges of G in a depth-first-search order continuing until
all edges are generated.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --A directed/undirected graph/multigraph.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.traversal.edgedfs.edge_dfs(G=<G_variable>)
