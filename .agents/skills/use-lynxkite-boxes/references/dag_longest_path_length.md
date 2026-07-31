**DAG longest path length:**
Returns the longest path length in a DAG
parameters:
  - weight: str | None = weight --Edge data key to use for weight
  - default_weight: int | None = 1 --The weight of edges that do not have a weight attribute
  - G: <class 'networkx.classes.graph.Graph'> = ? --A directed acyclic graph (DAG)
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.dag.dag_longest_path_length(weight=<weight_value>, default_weight=<default_weight_value>, G=<G_variable>)
