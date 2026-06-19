---
name: networkx-linalg-spectrum
description: Collection of operations - Laplacian spectrum, Adjacency spectrum, Modularity spectrum, Normalized Laplacian spectrum, Bethe–Hessian spectrum
---

**Laplacian spectrum:**
Returns eigenvalues of the Laplacian of G
parameters:
  - weight: str | None = weight --The edge data key used to compute each value in the matrix.
If None, then each edge has weight 1.
  - G: <class 'networkx.classes.graph.Graph'> = ? --A NetworkX graph

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.linalg.spectrum.laplacian_spectrum(weight=<weight_value>, G=<G_variable>)

**Adjacency spectrum:**
Returns eigenvalues of the adjacency matrix of G.
parameters:
  - weight: str | None = weight --The edge data key used to compute each value in the matrix.
If None, then each edge has weight 1.
  - G: <class 'networkx.classes.graph.Graph'> = ? --A NetworkX graph

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.linalg.spectrum.adjacency_spectrum(weight=<weight_value>, G=<G_variable>)

**Modularity spectrum:**
Returns eigenvalues of the modularity matrix of G.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --A NetworkX Graph or DiGraph

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.linalg.spectrum.modularity_spectrum(G=<G_variable>)

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

**Bethe–Hessian spectrum:**
Returns eigenvalues of the Bethe Hessian matrix of G.
parameters:
  - r: <class 'float'> = ? --Regularizer parameter
  - G: <class 'networkx.classes.graph.Graph'> = ? --A NetworkX Graph or DiGraph

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.linalg.spectrum.bethe_hessian_spectrum(r=<r_value>, G=<G_variable>)
