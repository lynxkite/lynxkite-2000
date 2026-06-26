**Subgraph centrality:**
Returns subgraph centrality for each node in G.

Subgraph centrality  of a node `n` is the sum of weighted closed
walks of all lengths starting and ending at node `n`. The weights
decrease with path length. Each closed walk is associated with a
connected subgraph ([1]_).
parameters:
  - normalized: <class 'bool'> = ? --If True, normalize the centrality values using the largest eigenvalue of the
adjacency matrix so that the centrality values are generally between 0 and 1.
  - G: <class 'networkx.classes.graph.Graph'> = ? --?

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.centrality.subgraph_alg.subgraph_centrality(normalized=<normalized_value>, G=<G_variable>)
