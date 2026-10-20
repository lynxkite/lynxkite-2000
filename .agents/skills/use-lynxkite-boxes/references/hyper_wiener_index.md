**Hyper wiener index:**
Returns the Hyper-Wiener index of the graph `G`.

The Hyper-Wiener index of a connected graph `G` is defined as

.. math::
    WW(G) = \frac{1}{2} \sum_{u,v \in V(G)} (d(u,v) + d(u,v)^2)

where ``d(u, v)`` is the shortest-path distance between nodes ``u`` and ``v``.
parameters:
  - weight: str | None = ? --The edge attribute to use for calculating shortest-path distances.
If None, all edges are considered to have a weight of 1.
  - G: <class 'networkx.classes.graph.Graph'> = ? --An undirected, connected graph.
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.wiener.hyper_wiener_index(weight=<weight_value>, G=<G_variable>)
