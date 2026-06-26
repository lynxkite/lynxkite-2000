**General random intersection graph:**
Returns a random intersection graph with independent probabilities
for connections between node and attribute sets.
parameters:
  - n: <class 'int'> = ? --The number of nodes in the first bipartite set (nodes)
  - m: <class 'int'> = ? --The number of nodes in the second bipartite set (attributes)
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.intersection.general_random_intersection_graph(n=<n_value>, m=<m_value>, seed=<seed_value>)
