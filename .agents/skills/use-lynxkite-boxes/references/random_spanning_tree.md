**Random spanning tree:**
Sample a random spanning tree using the edges weights of `G`.

This function supports two different methods for determining the
probability of the graph. If ``multiplicative=True``, the probability
is based on the product of edge weights, and if ``multiplicative=False``
it is based on the sum of the edge weight. However, since it is
easier to determine the total weight of all spanning trees for the
multiplicative version, that is significantly faster and should be used if
possible. Additionally, setting `weight` to `None` will cause a spanning tree
to be selected with uniform probability.

The function uses algorithm A8 in [1]_ .
parameters:
  - weight: <class 'str'> = ? --The edge key for the edge attribute holding edge weight.
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.
  - G: <class 'networkx.classes.graph.Graph'> = ? --An undirected version of the original graph.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.tree.mst.random_spanning_tree(weight=<weight_value>, seed=<seed_value>, G=<G_variable>)
