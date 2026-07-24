**Average clustering:**
Compute the average clustering coefficient for the graph G.

The clustering coefficient for the graph is the average,

.. math::

   C = \frac{1}{n}\sum_{v \in G} c_v,

where :math:`n` is the number of nodes in `G`.
parameters:
  - weight: str | None = ? --The edge attribute that holds the numerical value used as a weight.
If None, then each edge has weight 1.
  - count_zeros: <class 'bool'> = ? --If False include only the nodes with nonzero clustering in the average.
  - G: <class 'networkx.classes.graph.Graph'> = ? --?
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.cluster.average_clustering(weight=<weight_value>, count_zeros=<count_zeros_value>, G=<G_variable>)
