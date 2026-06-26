**DAG longest path:**
Returns the longest path in a directed acyclic graph (DAG).

If `G` has edges with `weight` attribute the edge data are used as
weight values.
parameters:
  - weight: str | None = weight --Edge data key to use for weight
  - default_weight: int | None = 1 --The weight of edges that do not have a weight attribute
  - G: <class 'networkx.classes.graph.Graph'> = ? --A directed acyclic graph (DAG)

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.dag.dag_longest_path(weight=<weight_value>, default_weight=<default_weight_value>, G=<G_variable>)
