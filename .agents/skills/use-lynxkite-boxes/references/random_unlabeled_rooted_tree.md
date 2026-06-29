**Random unlabeled rooted tree:**
Returns a number of unlabeled rooted trees uniformly at random

Returns one or more (depending on `number_of_trees`)
unlabeled rooted trees with `n` nodes drawn uniformly
at random.
parameters:
  - n: <class 'int'> = ? --The number of nodes
  - number_of_trees: int | None = ? --If not None, this number of trees is generated and returned.
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.trees.random_unlabeled_rooted_tree(n=<n_value>, number_of_trees=<number_of_trees_value>, seed=<seed_value>)
