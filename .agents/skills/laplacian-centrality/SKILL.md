---
name: laplacian-centrality
description: Compute the Laplacian centrality for nodes in the graph `G`.
---

**Laplacian centrality:**
Compute the Laplacian centrality for nodes in the graph `G`.

The Laplacian Centrality of a node ``i`` is measured by the drop in the
Laplacian Energy after deleting node ``i`` from the graph. The Laplacian Energy
is the sum of the squared eigenvalues of a graph's Laplacian matrix.

.. math::

    C_L(u_i,G) = \frac{(\Delta E)_i}{E_L (G)} = \frac{E_L (G)-E_L (G_i)}{E_L (G)}

    E_L (G) = \sum_{i=0}^n \lambda_i^2

Where $E_L (G)$ is the Laplacian energy of graph `G`,
E_L (G_i) is the Laplacian energy of graph `G` after deleting node ``i``
and $\lambda_i$ are the eigenvalues of `G`'s Laplacian matrix.
This formula shows the normalized value. Without normalization,
the numerator on the right side is returned.
parameters:
  - normalized: <class 'bool'> = None -
  - weight: str | None = weight -
  - walk_type: str | None = None -
  - alpha: <class 'float'> = 0.95 -
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.centrality.laplacian.laplacian_centrality(normalized=<normalized_value>, weight=<weight_value>, walk_type=<walk_type_value>, alpha=<alpha_value>, G=<G_variable>)
