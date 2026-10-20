**Is negatively weighted:**
Returns True if `G` has negatively weighted edges.
parameters:
  - weight: str | None = weight --The attribute name used to query for edge weights.
  - G: <class 'networkx.classes.graph.Graph'> = ? --A NetworkX graph.
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.classes.function.is_negatively_weighted(weight=<weight_value>, G=<G_variable>)
