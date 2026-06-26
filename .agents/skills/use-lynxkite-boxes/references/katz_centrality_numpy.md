**Katz centrality NumPy:**
Compute the Katz centrality for the graph G.

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
  - alpha: <class 'float'> = 0.1 --Attenuation factor
  - beta: float | None = 1.0 --Weight attributed to the immediate neighborhood. If not a scalar the
dictionary must have an value for every node.
  - normalized: <class 'bool'> = ? --If True normalize the resulting values.
  - weight: str | None = ? --If None, all edge weights are considered equal.
Otherwise holds the name of the edge attribute used as weight.
In this measure the weight is interpreted as the connection strength.
  - G: <class 'networkx.classes.graph.Graph'> = ? --A NetworkX graph

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.centrality.katz.katz_centrality_numpy(alpha=<alpha_value>, beta=<beta_value>, normalized=<normalized_value>, weight=<weight_value>, G=<G_variable>)
