**Minimum spanning edges:**
Generate edges in a minimum spanning forest of an undirected
weighted graph.

A minimum spanning tree is a subgraph of the graph (a tree)
with the minimum sum of edge weights.  A spanning forest is a
union of the spanning trees for each connected component of the graph.
parameters:
  - algorithm: <class 'str'> = kruskal --The algorithm to use when finding a minimum spanning tree. Valid
choices are 'kruskal', 'prim', or 'boruvka'. The default is 'kruskal'.
  - weight: <class 'str'> = weight --Edge data key to use for weight (default 'weight').
  - keys: <class 'bool'> = ? --Whether to yield edge key in multigraphs in addition to the edge.
If `G` is not a multigraph, this is ignored.
  - data: bool | None = ? --If True yield the edge data along with the edge.
  - ignore_nan: <class 'bool'> = ? --If a NaN is found as an edge weight normally an exception is raised.
If `ignore_nan is True` then that edge is ignored instead.
  - G: <class 'networkx.classes.graph.Graph'> = ? --An undirected graph. If `G` is connected, then the algorithm finds a
spanning tree. Otherwise, a spanning forest is found.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.tree.mst.minimum_spanning_edges(algorithm=<algorithm_value>, weight=<weight_value>, keys=<keys_value>, data=<data_value>, ignore_nan=<ignore_nan_value>, G=<G_variable>)
