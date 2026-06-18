---
name: networkx-algorithms-regular
description: Collection of operations - Is regular, K factor
---

**Is regular:**
Determines whether a graph is regular.

A regular graph is a graph where all nodes have the same degree. A regular
digraph is a graph where all nodes have the same indegree and all nodes
have the same outdegree.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.regular.is_regular(G=<G_variable>)

**K factor:**
Compute a `k`-factor of a graph.

A `k`-factor of a graph is a spanning `k`-regular subgraph.
A spanning `k`-regular subgraph of `G` is a subgraph that contains
each node of `G` and a subset of the edges of `G` such that each
node has degree `k`.
parameters:
  - k: <class 'int'> = None -
  - matching_weight: str | None = weight -
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.regular.k_factor(k=<k_value>, matching_weight=<matching_weight_value>, G=<G_variable>)
