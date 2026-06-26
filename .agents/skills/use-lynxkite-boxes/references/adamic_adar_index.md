**Adamic–Adar index:**
Compute the Adamic-Adar index of all node pairs in ebunch.

Adamic-Adar index of `u` and `v` is defined as

.. math::

    \sum_{w \in \Gamma(u) \cap \Gamma(v)} \frac{1}{\log |\Gamma(w)|}

where $\Gamma(u)$ denotes the set of neighbors of $u$.
This index leads to zero-division for nodes only connected via self-loops.
It is intended to be used when no self-loops are present.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --NetworkX undirected graph.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.link_prediction.adamic_adar_index(G=<G_variable>)
