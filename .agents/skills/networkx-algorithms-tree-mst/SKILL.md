---
name: networkx-algorithms-tree-mst
description: Collection of operations - Minimum spanning edges, Maximum spanning edges, Minimum spanning tree, Maximum spanning tree, Random spanning tree, Partition spanning tree
---

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

**Maximum spanning edges:**
Generate edges in a maximum spanning forest of an undirected
weighted graph.

A maximum spanning tree is a subgraph of the graph (a tree)
with the maximum possible sum of edge weights.  A spanning forest is a
union of the spanning trees for each connected component of the graph.
parameters:
  - algorithm: <class 'str'> = kruskal --The algorithm to use when finding a maximum spanning tree. Valid
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
output_variable = networkx.algorithms.tree.mst.maximum_spanning_edges(algorithm=<algorithm_value>, weight=<weight_value>, keys=<keys_value>, data=<data_value>, ignore_nan=<ignore_nan_value>, G=<G_variable>)

**Minimum spanning tree:**
Returns a minimum spanning tree or forest on an undirected graph `G`.
parameters:
  - weight: <class 'str'> = weight --Data key to use for edge weights.
  - algorithm: <class 'str'> = kruskal --The algorithm to use when finding a minimum spanning tree. Valid
choices are 'kruskal', 'prim', or 'boruvka'. The default is
'kruskal'.
  - ignore_nan: <class 'bool'> = ? --If a NaN is found as an edge weight normally an exception is raised.
If `ignore_nan is True` then that edge is ignored instead.
  - G: <class 'networkx.classes.graph.Graph'> = ? --An undirected graph. If `G` is connected, then the algorithm finds a
spanning tree. Otherwise, a spanning forest is found.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.tree.mst.minimum_spanning_tree(weight=<weight_value>, algorithm=<algorithm_value>, ignore_nan=<ignore_nan_value>, G=<G_variable>)

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

**Random spanning tree:**
Sample a random spanning tree using the edges weights of `G`.

This function supports two different methods for determining the
probability of the graph. If ``multiplicative=True``, the probability
is based on the product of edge weights, and if ``multiplicative=False``
it is based on the sum of the edge weight. However, since it is
easier to determine the total weight of all spanning trees for the
multiplicative version, that is significantly faster and should be used if
possible. Additionally, setting `weight` to `None` will cause a spanning tree
to be selected with uniform probability.

The function uses algorithm A8 in [1]_ .
parameters:
  - weight: <class 'str'> = ? --The edge key for the edge attribute holding edge weight.
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.
  - G: <class 'networkx.classes.graph.Graph'> = ? --An undirected version of the original graph.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.tree.mst.random_spanning_tree(weight=<weight_value>, seed=<seed_value>, G=<G_variable>)

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
