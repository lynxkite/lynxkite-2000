**Barycenter:**
Calculate barycenter of a connected graph, optionally with edge weights.

The :dfn:`barycenter` a
:func:`connected <networkx.algorithms.components.is_connected>` graph
:math:`G` is the subgraph induced by the set of its nodes :math:`v`
minimizing the objective function

.. math::

    \sum_{u \in V(G)} d_G(u, v),

where :math:`d_G` is the (possibly weighted) :func:`path length
<networkx.algorithms.shortest_paths.generic.shortest_path_length>`.
The barycenter is also called the :dfn:`median`. See [West01]_, p. 78.
parameters:
  - weight: str | None = ? --Passed through to
:func:`~networkx.algorithms.shortest_paths.generic.shortest_path_length`.
  - G: <class 'networkx.classes.graph.Graph'> = ? --The connected graph :math:`G`.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.distance_measures.barycenter(weight=<weight_value>, G=<G_variable>)
