**Normalized Laplacian matrix:**
Returns the normalized Laplacian matrix of G.

The normalized graph Laplacian is the matrix

.. math::

    N = D^{-1/2} L D^{-1/2}

where `L` is the graph Laplacian and `D` is the diagonal matrix of
node degrees [1]_.
parameters:
  - weight: str | None = weight --The edge data key used to compute each value in the matrix.
If None, then each edge has weight 1.
  - G: <class 'networkx.classes.graph.Graph'> = ? --A NetworkX graph

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.linalg.laplacianmatrix.normalized_laplacian_matrix(weight=<weight_value>, G=<G_variable>)
