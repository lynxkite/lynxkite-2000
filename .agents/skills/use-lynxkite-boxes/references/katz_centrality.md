**Katz centrality:**
Compute the Katz centrality for the nodes of the graph G.

Katz centrality computes the centrality for a node based on the centrality
of its neighbors. It is a generalization of the eigenvector centrality. The
Katz centrality for node $i$ is

.. math::

    x_i = \alpha \sum_{j} A_{ij} x_j + \beta,

where $A$ is the adjacency matrix of graph G with eigenvalues $\lambda$.

The parameter $\beta$ controls the initial centrality and

.. math::

    \alpha < \frac{1}{\lambda_{\max}}.

Katz centrality computes the relative influence of a node within a
network by measuring the number of the immediate neighbors (first
degree nodes) and also all other nodes in the network that connect
to the node under consideration through these immediate neighbors.

Extra weight can be provided to immediate neighbors through the
parameter $\beta$.  Connections made with distant neighbors
are, however, penalized by an attenuation factor $\alpha$ which
should be strictly less than the inverse largest eigenvalue of the
adjacency matrix in order for the Katz centrality to be computed
correctly. More information is provided in [1]_.
parameters:
  - alpha: float | None = 0.1 --Attenuation factor
  - beta: float | None = 1.0 --Weight attributed to the immediate neighborhood. If not a scalar, the
dictionary must have a value for every node.
  - max_iter: int | None = 1000 --Maximum number of iterations in power method.
  - tol: float | None = 1e-06 --Error tolerance used to check convergence in power method iteration.
  - normalized: bool | None = ? --If True normalize the resulting values.
  - weight: str | None = ? --If None, all edge weights are considered equal.
Otherwise holds the name of the edge attribute used as weight.
In this measure the weight is interpreted as the connection strength.
  - G: <class 'networkx.classes.graph.Graph'> = ? --A NetworkX graph.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.centrality.katz.katz_centrality(alpha=<alpha_value>, beta=<beta_value>, max_iter=<max_iter_value>, tol=<tol_value>, normalized=<normalized_value>, weight=<weight_value>, G=<G_variable>)
