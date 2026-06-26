**Normalized Laplacian spectrum:**
Return eigenvalues of the normalized Laplacian of G
parameters:
  - weight: str | None = weight --The edge data key used to compute each value in the matrix.
If None, then each edge has weight 1.
  - G: <class 'networkx.classes.graph.Graph'> = ? --A NetworkX graph

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.linalg.spectrum.normalized_laplacian_spectrum(weight=<weight_value>, G=<G_variable>)
