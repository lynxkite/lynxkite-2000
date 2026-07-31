**Max weight matching:**
Compute a maximum-weighted matching of G.

A matching is a subset of edges in which no node occurs more than once.
The weight of a matching is the sum of the weights of its edges.
A maximal matching cannot add more edges and still be a matching.
The cardinality of a matching is the number of matched edges.
parameters:
  - maxcardinality: bool | None = ? --If maxcardinality is True, compute the maximum-cardinality matching
with maximum weight among all maximum-cardinality matchings.
  - weight: str | None = weight --Edge data key corresponding to the edge weight.
If key not found, uses 1 as weight.
  - G: <class 'networkx.classes.graph.Graph'> = ? --Undirected graph
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.matching.max_weight_matching(maxcardinality=<maxcardinality_value>, weight=<weight_value>, G=<G_variable>)
