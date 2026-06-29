**Power-law cluster graph:**
Holme and Kim algorithm for growing graphs with powerlaw
degree distribution and approximate average clustering.
parameters:
  - n: <class 'int'> = ? --the number of nodes
  - m: <class 'int'> = ? --the number of random edges to add for each new node
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.random_graphs.powerlaw_cluster_graph(n=<n_value>, m=<m_value>, seed=<seed_value>)
