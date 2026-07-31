**Random labeled rooted tree:**
Returns a labeled rooted tree with `n` nodes.

The returned tree is chosen uniformly at random from all labeled rooted trees.
parameters:
  - n: <class 'int'> = ? --The number of nodes
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.generators.trees.random_labeled_rooted_tree(n=<n_value>, seed=<seed_value>)
