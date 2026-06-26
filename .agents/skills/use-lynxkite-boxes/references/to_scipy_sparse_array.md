**To SciPy sparse array:**
Returns the graph adjacency matrix as a SciPy sparse array.
parameters:
  - weight: str | None = weight --The edge attribute that holds the numerical value used for
the edge weight.  If None then all edge weights are 1.
  - G: <class 'networkx.classes.graph.Graph'> = ? --The NetworkX graph used to construct the sparse array.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.convert_matrix.to_scipy_sparse_array(weight=<weight_value>, G=<G_variable>)
