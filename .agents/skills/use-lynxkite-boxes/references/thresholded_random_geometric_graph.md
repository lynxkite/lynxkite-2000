**Thresholded random geometric graph:**
Returns a thresholded random geometric graph in the unit cube.

The thresholded random geometric graph [1] model places `n` nodes
uniformly at random in the unit cube of dimensions `dim`. Each node
`u` is assigned a weight :math:`w_u`. Two nodes `u` and `v` are
joined by an edge if they are within the maximum connection distance,
`radius` computed by the `p`-Minkowski distance and the summation of
weights :math:`w_u` + :math:`w_v` is greater than or equal
to the threshold parameter `theta`.

Edges within `radius` of each other are determined using a KDTree when
SciPy is available. This reduces the time complexity from :math:`O(n^2)`
to :math:`O(n)`.
parameters:
  - radius: <class 'float'> = ? --Distance threshold value
  - theta: <class 'float'> = ? --Threshold value
  - dim: int | None = 2 --Dimension of graph
  - p: float | None = 2 --Which Minkowski distance metric to use.  `p` has to meet the condition
``1 <= p <= infinity``.

If this argument is not specified, the :math:`L^2` metric
(the Euclidean distance metric), p = 2 is used.

This should not be confused with the `p` of an Erdős-Rényi random
graph, which represents probability.
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.geometric.thresholded_random_geometric_graph(radius=<radius_value>, theta=<theta_value>, dim=<dim_value>, p=<p_value>, seed=<seed_value>)
