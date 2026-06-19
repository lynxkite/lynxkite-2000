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
