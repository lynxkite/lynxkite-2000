**Eigenvector centrality NumPy:**
Compute the eigenvector centrality for the graph `G`.

Eigenvector centrality computes the centrality for a node by adding
the centrality of its predecessors. The centrality for node $i$ is the
$i$-th element of a left eigenvector associated with the eigenvalue $\lambda$
of maximum modulus that is positive. Such an eigenvector $x$ is
defined up to a multiplicative constant by the equation

.. math::

     \lambda x^T = x^T A,

where $A$ is the adjacency matrix of the graph `G`. By definition of
row-column product, the equation above is equivalent to

.. math::

    \lambda x_i = \sum_{j\to i}x_j.

That is, adding the eigenvector centralities of the predecessors of
$i$ one obtains the eigenvector centrality of $i$ multiplied by
$\lambda$. In the case of undirected graphs, $x$ also solves the familiar
right-eigenvector equation $Ax = \lambda x$.

By virtue of the Perron--Frobenius theorem [1]_, if `G` is (strongly)
connected, there is a unique eigenvector $x$, and all its entries
are strictly positive.

However, if `G` is not (strongly) connected, there might be several left
eigenvectors associated with $\lambda$, and some of their elements
might be zero.
Depending on the method used to choose eigenvectors, round-off error can affect
which of the infinitely many eigenvectors is reported.
This can lead to inconsistent results for the same graph,
which the underlying implementation is not robust to.
For this reason, only (strongly) connected graphs are accepted.
parameters:
  - weight: str | None = ? --If ``None``, all edge weights are considered equal. Otherwise holds the
name of the edge attribute used as weight. In this measure the
weight is interpreted as the connection strength.
  - max_iter: int | None = 50 --Maximum number of Arnoldi update iterations allowed.
  - tol: float | None = 0 --Relative accuracy for eigenvalues (stopping criterion).
The default value of 0 implies machine precision.
  - G: <class 'networkx.classes.graph.Graph'> = ? --A connected NetworkX graph.
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.centrality.eigenvector.eigenvector_centrality_numpy(weight=<weight_value>, max_iter=<max_iter_value>, tol=<tol_value>, G=<G_variable>)
