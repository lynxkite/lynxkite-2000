**Constraint:**
Returns the constraint on all nodes in the graph ``G``.

The *constraint* is a measure of the extent to which a node *v* is
invested in those nodes that are themselves invested in the
neighbors of *v*. Formally, the *constraint on v*, denoted `c(v)`,
is defined by

.. math::

   c(v) = \sum_{w \in N(v) \setminus \{v\}} \ell(v, w)

where $N(v)$ is the subset of the neighbors of `v` that are either
predecessors or successors of `v` and $\ell(v, w)$ is the local
constraint on `v` with respect to `w` [1]_. For the definition of local
constraint, see :func:`local_constraint`.
parameters:
  - weight: str | None = ? --If None, all edge weights are considered equal.
Otherwise holds the name of the edge attribute used as weight.
  - G: <class 'networkx.classes.graph.Graph'> = ? --The graph containing ``v``. This can be either directed or undirected.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.structuralholes.constraint(weight=<weight_value>, G=<G_variable>)
