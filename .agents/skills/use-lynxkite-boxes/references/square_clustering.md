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
