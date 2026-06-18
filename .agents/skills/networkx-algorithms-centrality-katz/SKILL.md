---
name: networkx-algorithms-centrality-katz
description: Collection of operations - Katz centrality, Katz centrality NumPy
---

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
  - alpha: float | None = 0.1 -
  - beta: float | None = 1.0 -
  - max_iter: int | None = 1000 -
  - tol: float | None = 1e-06 -
  - normalized: bool | None = None -
  - weight: str | None = None -
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.centrality.katz.katz_centrality(alpha=<alpha_value>, beta=<beta_value>, max_iter=<max_iter_value>, tol=<tol_value>, normalized=<normalized_value>, weight=<weight_value>, G=<G_variable>)

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
  - alpha: <class 'float'> = 0.1 -
  - beta: float | None = 1.0 -
  - normalized: <class 'bool'> = None -
  - weight: str | None = None -
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.centrality.katz.katz_centrality_numpy(alpha=<alpha_value>, beta=<beta_value>, normalized=<normalized_value>, weight=<weight_value>, G=<G_variable>)
