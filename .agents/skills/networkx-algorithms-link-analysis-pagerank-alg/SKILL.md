---
name: networkx-algorithms-link-analysis-pagerank-alg
description: Collection of operations - PageRank, Google matrix
---

**PageRank:**
Returns the PageRank of the nodes in the graph.

PageRank computes a ranking of the nodes in the graph G based on
the structure of the incoming links. It was originally designed as
an algorithm to rank web pages.
parameters:
  - alpha: float | None = 0.85 --Damping parameter for PageRank, default=0.85.
  - max_iter: int | None = 100 --Maximum number of iterations in power method eigenvalue solver.
  - tol: float | None = 1e-06 --Error tolerance used to check convergence in power method solver.
The iteration will stop after a tolerance of ``len(G) * tol`` is reached.
  - weight: str | None = weight --Edge data key to use as weight.  If None weights are set to 1.
  - G: <class 'networkx.classes.graph.Graph'> = ? --A NetworkX graph.  Undirected graphs will be converted to a directed
graph with two directed edges for each undirected edge.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.link_analysis.pagerank_alg.pagerank(alpha=<alpha_value>, max_iter=<max_iter_value>, tol=<tol_value>, weight=<weight_value>, G=<G_variable>)

**Google matrix:**
Returns the Google matrix of the graph.
parameters:
  - alpha: <class 'float'> = 0.85 --The damping factor.
  - weight: str | None = weight --Edge data key to use as weight.  If None weights are set to 1.
  - G: <class 'networkx.classes.graph.Graph'> = ? --A NetworkX graph.  Undirected graphs will be converted to a directed
graph with two directed edges for each undirected edge.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.link_analysis.pagerank_alg.google_matrix(alpha=<alpha_value>, weight=<weight_value>, G=<G_variable>)
