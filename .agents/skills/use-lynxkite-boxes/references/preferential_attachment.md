**Preferential attachment:**
Compute the preferential attachment score of all node pairs in ebunch.

Preferential attachment score of `u` and `v` is defined as

.. math::

    |\Gamma(u)| |\Gamma(v)|

where $\Gamma(u)$ denotes the set of neighbors of $u$.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --NetworkX undirected graph.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.link_prediction.preferential_attachment(G=<G_variable>)
