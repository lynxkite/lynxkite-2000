**Uniform random intersection graph:**
Returns a uniform random intersection graph.
parameters:
  - n: <class 'int'> = ? --The number of nodes in the first bipartite set (nodes)
  - m: <class 'int'> = ? --The number of nodes in the second bipartite set (attributes)
  - p: <class 'float'> = ? --Probability of connecting nodes between bipartite sets
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.intersection.uniform_random_intersection_graph(n=<n_value>, m=<m_value>, p=<p_value>, seed=<seed_value>)
