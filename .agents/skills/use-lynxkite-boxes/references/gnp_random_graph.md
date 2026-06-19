**Binomial graph:**
Returns a $G_{n,p}$ random graph, also known as an Erdős-Rényi graph
or a binomial graph.

The $G_{n,p}$ model chooses each of the possible edges with probability $p$.
parameters:
  - n: <class 'int'> = ? --The number of nodes.
  - p: <class 'float'> = ? --Probability for edge creation.
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.
  - directed: bool | None = ? --If True, this function returns a directed graph.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.random_graphs.gnp_random_graph(n=<n_value>, p=<p_value>, seed=<seed_value>, directed=<directed_value>)
