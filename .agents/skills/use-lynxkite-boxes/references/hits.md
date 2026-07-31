**Hits:**
Returns HITS hubs and authorities values for nodes.

The HITS algorithm computes two numbers for a node.
Authorities estimates the node value based on the incoming links.
Hubs estimates the node value based on outgoing links.
parameters:
  - max_iter: int | None = 100 --Maximum number of iterations in power method.
  - tol: float | None = 1e-08 --Error tolerance used to check convergence in power method iteration.
  - normalized: <class 'bool'> = ? --Normalize results by the sum of all of the values.
  - G: <class 'networkx.classes.graph.Graph'> = ? --A NetworkX graph
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.link_analysis.hits_alg.hits(max_iter=<max_iter_value>, tol=<tol_value>, normalized=<normalized_value>, G=<G_variable>)
