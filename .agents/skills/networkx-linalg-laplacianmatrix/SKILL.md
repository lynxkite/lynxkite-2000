---
name: networkx-linalg-laplacianmatrix
description: Collection of operations - Laplacian matrix, Normalized Laplacian matrix, Directed Laplacian matrix, Directed combinatorial Laplacian matrix
---

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

**Directed Laplacian matrix:**
Returns the directed Laplacian matrix of G.

The graph directed Laplacian is the matrix

.. math::

    L = I - \frac{1}{2} \left (\Phi^{1/2} P \Phi^{-1/2} + \Phi^{-1/2} P^T \Phi^{1/2} \right )

where `I` is the identity matrix, `P` is the transition matrix of the
graph, and `\Phi` a matrix with the Perron vector of `P` in the diagonal and
zeros elsewhere [1]_.

Depending on the value of walk_type, `P` can be the transition matrix
induced by a random walk, a lazy random walk, or a random walk with
teleportation (PageRank).
parameters:
  - weight: str | None = weight --The edge data key used to compute each value in the matrix.
If None, then each edge has weight 1.
  - walk_type: str | None = ? --One of ``"random"``, ``"lazy"``, or ``"pagerank"``. If ``walk_type=None``
(the default), then a value is selected according to the properties of `G`:
- ``walk_type="random"`` if `G` is strongly connected and aperiodic
- ``walk_type="lazy"`` if `G` is strongly connected but not aperiodic
- ``walk_type="pagerank"`` for all other cases.
  - alpha: <class 'float'> = 0.95 --(1 - alpha) is the teleportation probability used with pagerank
  - G: <class 'networkx.classes.graph.Graph'> = ? --A NetworkX graph

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.linalg.laplacianmatrix.directed_laplacian_matrix(weight=<weight_value>, walk_type=<walk_type_value>, alpha=<alpha_value>, G=<G_variable>)

**Directed combinatorial Laplacian matrix:**
Return the directed combinatorial Laplacian matrix of G.

The graph directed combinatorial Laplacian is the matrix

.. math::

    L = \Phi - \frac{1}{2} \left (\Phi P + P^T \Phi \right)

where `P` is the transition matrix of the graph and `\Phi` a matrix
with the Perron vector of `P` in the diagonal and zeros elsewhere [1]_.

Depending on the value of walk_type, `P` can be the transition matrix
induced by a random walk, a lazy random walk, or a random walk with
teleportation (PageRank).
parameters:
  - weight: str | None = weight --The edge data key used to compute each value in the matrix.
If None, then each edge has weight 1.
  - walk_type: str | None = ? --One of ``"random"``, ``"lazy"``, or ``"pagerank"``. If ``walk_type=None``
(the default), then a value is selected according to the properties of `G`:
- ``walk_type="random"`` if `G` is strongly connected and aperiodic
- ``walk_type="lazy"`` if `G` is strongly connected but not aperiodic
- ``walk_type="pagerank"`` for all other cases.
  - alpha: <class 'float'> = 0.95 --(1 - alpha) is the teleportation probability used with pagerank
  - G: <class 'networkx.classes.graph.Graph'> = ? --A NetworkX graph

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.linalg.laplacianmatrix.directed_combinatorial_laplacian_matrix(weight=<weight_value>, walk_type=<walk_type_value>, alpha=<alpha_value>, G=<G_variable>)
