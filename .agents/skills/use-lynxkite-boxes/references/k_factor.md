**K factor:**
Compute a `k`-factor of a graph.

A `k`-factor of a graph is a spanning `k`-regular subgraph.
A spanning `k`-regular subgraph of `G` is a subgraph that contains
each node of `G` and a subset of the edges of `G` such that each
node has degree `k`.
parameters:
  - k: <class 'int'> = ? --The degree of the `k`-factor.
  - matching_weight: str | None = weight --Edge attribute name corresponding to the edge weight.
If not present, the edge is assumed to have weight 1.
Used for finding the max-weighted perfect matching.
  - G: <class 'networkx.classes.graph.Graph'> = ? --An undirected graph.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.regular.k_factor(k=<k_value>, matching_weight=<matching_weight_value>, G=<G_variable>)
