**Random power-law tree sequence:**
Returns a degree sequence for a tree with a power law distribution.
parameters:
  - gamma: <class 'float'> = 3 --Exponent of the power law.
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.
  - tries: <class 'int'> = 100 --Number of attempts to adjust the sequence to make it a tree.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.random_graphs.random_powerlaw_tree_sequence(gamma=<gamma_value>, seed=<seed_value>, tries=<tries_value>)
