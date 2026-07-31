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
