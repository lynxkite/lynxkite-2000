**To NumPy array:**
Returns the graph adjacency matrix as a NumPy array.
parameters:
  - weight: <class 'str'> = weight --The edge attribute that holds the numerical value used for
the edge weight. If an edge does not have that attribute, then the
value 1 is used instead. `weight` must be ``None`` if a structured
dtype is used.
  - G: <class 'networkx.classes.graph.Graph'> = ? --The NetworkX graph used to construct the NumPy array.
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.convert_matrix.to_numpy_array(weight=<weight_value>, G=<G_variable>)
