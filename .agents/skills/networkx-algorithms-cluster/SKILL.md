---
name: networkx-algorithms-cluster
description: Collection of operations - Triangles, All triangles, Average clustering, Clustering, Transitivity, Square clustering, Generalized degree
---

**Triangles:**
Compute the number of triangles.

Finds the number of triangles that include a node as one vertex.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --A networkx graph

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.cluster.triangles(G=<G_variable>)

**All triangles:**
Yields all unique triangles in an undirected graph.

A triangle is a set of three distinct nodes where each node is connected to
the other two.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --An undirected graph.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.cluster.all_triangles(G=<G_variable>)

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

**Transitivity:**
Compute graph transitivity, the fraction of all possible triangles
present in G.

Possible triangles are identified by the number of "triads"
(two edges with a shared vertex).

The transitivity is

.. math::

    T = 3\frac{\#triangles}{\#triads}.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --?

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.cluster.transitivity(G=<G_variable>)

**Square clustering:**
Compute the squares clustering coefficient for nodes.

For each node return the fraction of possible squares that exist at
the node [1]_

.. math::
   C_4(v) = \frac{ \sum_{u=1}^{k_v}
   \sum_{w=u+1}^{k_v} q_v(u,w) }{ \sum_{u=1}^{k_v}
   \sum_{w=u+1}^{k_v} [a_v(u,w) + q_v(u,w)]},

where :math:`q_v(u,w)` are the number of common neighbors of :math:`u` and
:math:`w` other than :math:`v` (ie squares), and :math:`a_v(u,w) = (k_u -
(1+q_v(u,w)+\theta_{uv})) + (k_w - (1+q_v(u,w)+\theta_{uw}))`, where
:math:`\theta_{uw} = 1` if :math:`u` and :math:`w` are connected and 0
otherwise. [2]_
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --?

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.cluster.square_clustering(G=<G_variable>)

**Generalized degree:**
Compute the generalized degree for nodes.

For each node, the generalized degree shows how many edges of given
triangle multiplicity the node is connected to. The triangle multiplicity
of an edge is the number of triangles an edge participates in. The
generalized degree of node :math:`i` can be written as a vector
:math:`\mathbf{k}_i=(k_i^{(0)}, \dotsc, k_i^{(N-2)})` where
:math:`k_i^{(j)}` is the number of edges attached to node :math:`i` that
participate in :math:`j` triangles.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --?

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.cluster.generalized_degree(G=<G_variable>)
