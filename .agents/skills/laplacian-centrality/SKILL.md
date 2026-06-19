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
  - normalized: <class 'bool'> = ? --If True the centrality score is scaled so the sum over all nodes is 1.
If False the centrality score for each node is the drop in Laplacian
energy when that node is removed.
  - weight: str | None = weight --Optional parameter `weight` to compute the Laplacian matrix.
The edge data key used to compute each value in the matrix.
If None, then each edge has weight 1.
  - walk_type: str | None = ? --Optional parameter `walk_type` used when calling
:func:`directed_laplacian_matrix <networkx.directed_laplacian_matrix>`.
One of ``"random"``, ``"lazy"``, or ``"pagerank"``. If ``walk_type=None``
(the default), then a value is selected according to the properties of `G`:
- ``walk_type="random"`` if `G` is strongly connected and aperiodic
- ``walk_type="lazy"`` if `G` is strongly connected but not aperiodic
- ``walk_type="pagerank"`` for all other cases.
  - alpha: <class 'float'> = 0.95 --Optional parameter `alpha` used when calling
:func:`directed_laplacian_matrix <networkx.directed_laplacian_matrix>`.
(1 - alpha) is the teleportation probability used with pagerank.
  - G: <class 'networkx.classes.graph.Graph'> = ? --A networkx graph

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.centrality.laplacian.laplacian_centrality(normalized=<normalized_value>, weight=<weight_value>, walk_type=<walk_type_value>, alpha=<alpha_value>, G=<G_variable>)
