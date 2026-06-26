**Jaccard coefficient:**
Compute the Jaccard coefficient of all node pairs in ebunch.

Jaccard coefficient of nodes `u` and `v` is defined as

.. math::

    \frac{|\Gamma(u) \cap \Gamma(v)|}{|\Gamma(u) \cup \Gamma(v)|}

where $\Gamma(u)$ denotes the set of neighbors of $u$.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --A NetworkX undirected graph.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.link_prediction.jaccard_coefficient(G=<G_variable>)
