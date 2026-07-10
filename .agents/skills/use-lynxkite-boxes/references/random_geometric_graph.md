**Random geometric graph:**
Returns a random geometric graph in the unit cube of dimensions `dim`.

The random geometric graph model places `n` nodes uniformly at
random in the unit cube. Two nodes are joined by an edge if the
distance between the nodes is at most `radius`.

Edges are determined using a KDTree when SciPy is available.
This reduces the time complexity from $O(n^2)$ to $O(n)$.
parameters:
  - radius: <class 'float'> = ? --Distance threshold value
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
  output_variable = networkx.generators.geometric.random_geometric_graph(radius=<radius_value>, dim=<dim_value>, p=<p_value>, seed=<seed_value>)
