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
  - algorithm: <class 'str'> = kruskal - .
  - weight: <class 'str'> = weight - .
  - keys: <class 'bool'> = None - .
  - data: bool | None = None - .
  - ignore_nan: <class 'bool'> = None - .
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.algorithms.tree.mst.minimum_spanning_edges(algorithm=<algorithm_value>, weight=<weight_value>, keys=<keys_value>, data=<data_value>, ignore_nan=<ignore_nan_value>, G=<G_variable>)

**Maximum spanning edges:**
Generate edges in a maximum spanning forest of an undirected
weighted graph.

A maximum spanning tree is a subgraph of the graph (a tree)
with the maximum possible sum of edge weights.  A spanning forest is a
union of the spanning trees for each connected component of the graph.
parameters:
  - algorithm: <class 'str'> = kruskal - .
  - weight: <class 'str'> = weight - .
  - keys: <class 'bool'> = None - .
  - data: bool | None = None - .
  - ignore_nan: <class 'bool'> = None - .
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.algorithms.tree.mst.maximum_spanning_edges(algorithm=<algorithm_value>, weight=<weight_value>, keys=<keys_value>, data=<data_value>, ignore_nan=<ignore_nan_value>, G=<G_variable>)

**Minimum spanning tree:**
Returns a minimum spanning tree or forest on an undirected graph `G`.
parameters:
  - weight: <class 'str'> = weight - .
  - algorithm: <class 'str'> = kruskal - .
  - ignore_nan: <class 'bool'> = None - .
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.algorithms.tree.mst.minimum_spanning_tree(weight=<weight_value>, algorithm=<algorithm_value>, ignore_nan=<ignore_nan_value>, G=<G_variable>)

**Maximum spanning tree:**
Returns a maximum spanning tree or forest on an undirected graph `G`.
parameters:
  - weight: <class 'str'> = weight - .
  - algorithm: <class 'str'> = kruskal - .
  - ignore_nan: <class 'bool'> = None - .
  - G: <class 'networkx.classes.graph.Graph'> = None - .

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
  - weight: <class 'str'> = None - .
  - seed: int | None = None - .
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.algorithms.tree.mst.random_spanning_tree(weight=<weight_value>, seed=<seed_value>, G=<G_variable>)

**Partition spanning tree:**
Find a spanning tree while respecting a partition of edges.

Edges can be flagged as either `INCLUDED` which are required to be in the
returned tree, `EXCLUDED`, which cannot be in the returned tree and `OPEN`.

This is used in the SpanningTreeIterator to create new partitions following
the algorithm of Sörensen and Janssens [1]_.
parameters:
  - minimum: <class 'bool'> = None - .
  - weight: <class 'str'> = weight - .
  - partition: <class 'str'> = partition - .
  - ignore_nan: <class 'bool'> = None - .
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.algorithms.tree.mst.partition_spanning_tree(minimum=<minimum_value>, weight=<weight_value>, partition=<partition_value>, ignore_nan=<ignore_nan_value>, G=<G_variable>)
