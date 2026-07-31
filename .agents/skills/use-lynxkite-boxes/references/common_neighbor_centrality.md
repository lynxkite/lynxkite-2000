**Common neighbor centrality:**
Return the CCPA score for each pair of nodes.

Compute the Common Neighbor and Centrality based Parameterized Algorithm(CCPA)
score of all node pairs in ebunch.

CCPA score of `u` and `v` is defined as

.. math::

    \alpha \cdot (|\Gamma (u){\cap }^{}\Gamma (v)|)+(1-\alpha )\cdot \frac{N}{{d}_{uv}}

where $\Gamma(u)$ denotes the set of neighbors of $u$, $\Gamma(v)$ denotes the
set of neighbors of $v$, $\alpha$ is  parameter varies between [0,1], $N$ denotes
total number of nodes in the Graph and ${d}_{uv}$ denotes shortest distance
between $u$ and $v$.

This algorithm is based on two vital properties of nodes, namely the number
of common neighbors and their centrality. Common neighbor refers to the common
nodes between two nodes. Centrality refers to the prestige that a node enjoys
in a network.

.. seealso::

    :func:`common_neighbors`
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --NetworkX undirected graph.
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.link_prediction.common_neighbor_centrality(G=<G_variable>)
