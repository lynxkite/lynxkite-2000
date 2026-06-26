**Partition spanning tree:**
Find a spanning tree while respecting a partition of edges.

Edges can be flagged as either `INCLUDED` which are required to be in the
returned tree, `EXCLUDED`, which cannot be in the returned tree and `OPEN`.

This is used in the SpanningTreeIterator to create new partitions following
the algorithm of Sörensen and Janssens [1]_.
parameters:
  - minimum: <class 'bool'> = ? --Determines whether the returned tree is the minimum spanning tree of
the partition of the maximum one.
  - weight: <class 'str'> = weight --Data key to use for edge weights.
  - partition: <class 'str'> = partition --The key for the edge attribute containing the partition
data on the graph. Edges can be included, excluded or open using the
`EdgePartition` enum.
  - ignore_nan: <class 'bool'> = ? --If a NaN is found as an edge weight normally an exception is raised.
If `ignore_nan is True` then that edge is ignored instead.
  - G: <class 'networkx.classes.graph.Graph'> = ? --An undirected graph.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.tree.mst.partition_spanning_tree(minimum=<minimum_value>, weight=<weight_value>, partition=<partition_value>, ignore_nan=<ignore_nan_value>, G=<G_variable>)
