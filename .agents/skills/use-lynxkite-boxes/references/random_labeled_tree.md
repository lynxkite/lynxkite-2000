**Random labeled tree:**
Returns a labeled tree on `n` nodes chosen uniformly at random.

Generating uniformly distributed random Prüfer sequences and
converting them into the corresponding trees is a straightforward
method of generating uniformly distributed random labeled trees.
This function implements this method.
parameters:
  - n: <class 'int'> = ? --The number of nodes, greater than zero.
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.generators.trees.random_labeled_tree(n=<n_value>, seed=<seed_value>)
