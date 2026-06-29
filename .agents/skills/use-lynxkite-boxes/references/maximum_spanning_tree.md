**Maximum spanning tree:**
Returns a maximum spanning tree or forest on an undirected graph `G`.
parameters:
  - weight: <class 'str'> = weight --Data key to use for edge weights.
  - algorithm: <class 'str'> = kruskal --The algorithm to use when finding a maximum spanning tree. Valid
choices are 'kruskal', 'prim', or 'boruvka'. The default is
'kruskal'.
  - ignore_nan: <class 'bool'> = ? --If a NaN is found as an edge weight normally an exception is raised.
If `ignore_nan is True` then that edge is ignored instead.
  - G: <class 'networkx.classes.graph.Graph'> = ? --An undirected graph. If `G` is connected, then the algorithm finds a
spanning tree. Otherwise, a spanning forest is found.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.tree.mst.maximum_spanning_tree(weight=<weight_value>, algorithm=<algorithm_value>, ignore_nan=<ignore_nan_value>, G=<G_variable>)
