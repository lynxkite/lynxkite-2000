**Resource allocation index:**
Compute the resource allocation index of all node pairs in ebunch.

Resource allocation index of `u` and `v` is defined as

.. math::

    \sum_{w \in \Gamma(u) \cap \Gamma(v)} \frac{1}{|\Gamma(w)|}

where $\Gamma(u)$ denotes the set of neighbors of $u$.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --A NetworkX undirected graph.
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.link_prediction.resource_allocation_index(G=<G_variable>)
