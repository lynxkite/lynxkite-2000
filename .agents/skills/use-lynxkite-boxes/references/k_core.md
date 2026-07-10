**k-core:**
Returns the k-core of G.

A k-core is a maximal subgraph that contains nodes of degree `k` or more.
parameters:
  - k: int | None = ? --The order of the core. If not specified return the main core.
  - G: <class 'networkx.classes.graph.Graph'> = ? --A graph or directed graph
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.core.k_core(k=<k_value>, G=<G_variable>)
