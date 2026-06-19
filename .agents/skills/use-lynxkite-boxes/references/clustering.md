**Clustering:**
Compute the clustering coefficient for nodes.

For unweighted graphs, the clustering of a node :math:`u`
is the fraction of possible triangles through that node that exist,

.. math::

  c_u = \frac{2 T(u)}{deg(u)(deg(u)-1)},

where :math:`T(u)` is the number of triangles through node :math:`u` and
:math:`deg(u)` is the degree of :math:`u`.

For weighted graphs, there are several ways to define clustering [1]_.
the one used here is defined
as the geometric average of the subgraph edge weights [2]_,

.. math::

   c_u = \frac{1}{deg(u)(deg(u)-1))}
         \sum_{vw} (\hat{w}_{uv} \hat{w}_{uw} \hat{w}_{vw})^{1/3}.

The edge weights :math:`\hat{w}_{uv}` are normalized by the maximum weight
in the network :math:`\hat{w}_{uv} = w_{uv}/\max(w)`.

The value of :math:`c_u` is assigned to 0 if :math:`deg(u) < 2`.

Additionally, this weighted definition has been generalized to support negative edge weights [3]_.

For directed graphs, the clustering is similarly defined as the fraction
of all possible directed triangles or geometric average of the subgraph
edge weights for unweighted and weighted directed graph respectively [4]_.

.. math::

   c_u = \frac{T(u)}{2(deg^{tot}(u)(deg^{tot}(u)-1) - 2deg^{\leftrightarrow}(u))},

where :math:`T(u)` is the number of directed triangles through node
:math:`u`, :math:`deg^{tot}(u)` is the sum of in degree and out degree of
:math:`u` and :math:`deg^{\leftrightarrow}(u)` is the reciprocal degree of
:math:`u`.
parameters:
  - weight: str | None = ? --The edge attribute that holds the numerical value used as a weight.
If None, then each edge has weight 1.
  - G: <class 'networkx.classes.graph.Graph'> = ? --?

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.cluster.clustering(weight=<weight_value>, G=<G_variable>)
