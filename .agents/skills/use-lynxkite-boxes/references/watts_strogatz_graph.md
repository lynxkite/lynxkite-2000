**Watts–Strogatz graph:**
Returns a Watts–Strogatz small-world graph.
parameters:
  - n: <class 'int'> = ? --The number of nodes
  - k: <class 'int'> = ? --Each node is joined with its `k` nearest neighbors in a ring
topology.
  - p: <class 'float'> = ? --The probability of rewiring each edge
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.generators.random_graphs.watts_strogatz_graph(n=<n_value>, k=<k_value>, p=<p_value>, seed=<seed_value>)
