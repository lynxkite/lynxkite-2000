**Laplacian matrix:**
Returns the Laplacian matrix of G.

The graph Laplacian is the matrix L = D - A, where
A is the adjacency matrix and D is the diagonal matrix of node degrees.
parameters:
  - weight: str | None = weight --The edge data key used to compute each value in the matrix.
If None, then each edge has weight 1.
  - G: <class 'networkx.classes.graph.Graph'> = ? --A NetworkX graph
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.linalg.laplacianmatrix.laplacian_matrix(weight=<weight_value>, G=<G_variable>)
