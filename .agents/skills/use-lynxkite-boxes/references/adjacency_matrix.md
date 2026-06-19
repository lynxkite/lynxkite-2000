**Adjacency matrix:**
Returns adjacency matrix of `G`.
parameters:
  - weight: str | None = weight --The edge data key used to provide each value in the matrix.
If None, then each edge has weight 1.
  - G: <class 'networkx.classes.graph.Graph'> = ? --A NetworkX graph

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.linalg.graphmatrix.adjacency_matrix(weight=<weight_value>, G=<G_variable>)
