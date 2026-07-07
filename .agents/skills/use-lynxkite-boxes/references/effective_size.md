**Effective size:**
Returns the effective size of all nodes in the graph ``G``.

The *effective size* of a node's ego network is based on the concept
of redundancy. A person's ego network has redundancy to the extent
that her contacts are connected to each other as well. The
nonredundant part of a person's relationships is the effective
size of her ego network [1]_.  Formally, the effective size of a
node $u$, denoted $e(u)$, is defined by

.. math::

   e(u) = \sum_{v \in N(u) \setminus \{u\}}
   \left(1 - \sum_{w \in N(v)} p_{uw} m_{vw}\right)

where $N(u)$ is the set of neighbors of $u$ and $p_{uw}$ is the
normalized mutual weight of the (directed or undirected) edges
joining $u$ and $v$, for each vertex $u$ and $v$ [1]_. And $m_{vw}$
is the mutual weight of $v$ and $w$ divided by $v$ highest mutual
weight with any of its neighbors. The *mutual weight* of $u$ and $v$
is the sum of the weights of edges joining them (edge weights are
assumed to be one if the graph is unweighted).

For the case of unweighted and undirected graphs, Borgatti proposed
a simplified formula to compute effective size [2]_

.. math::

   e(u) = n - \frac{2t}{n}

where `t` is the number of ties in the ego network (not including
ties to ego) and `n` is the number of nodes (excluding ego).
parameters:
  - weight: str | None = ? --If None, all edge weights are considered equal.
Otherwise holds the name of the edge attribute used as weight.
  - G: <class 'networkx.classes.graph.Graph'> = ? --The graph containing ``v``. Directed graphs are treated like
undirected graphs when computing neighbors of ``v``.
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.structuralholes.effective_size(weight=<weight_value>, G=<G_variable>)
