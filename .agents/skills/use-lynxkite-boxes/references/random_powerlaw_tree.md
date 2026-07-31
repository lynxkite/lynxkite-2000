**Random power-law tree:**
Returns a tree with a power law degree distribution.
parameters:
  - n: <class 'int'> = ? --The number of nodes.
  - gamma: <class 'float'> = 3 --Exponent of the power law.
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.
  - tries: <class 'int'> = 100 --Number of attempts to adjust the sequence to make it a tree.
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.generators.random_graphs.random_powerlaw_tree(n=<n_value>, gamma=<gamma_value>, seed=<seed_value>, tries=<tries_value>)
