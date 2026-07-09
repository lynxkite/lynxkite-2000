**Random unlabeled rooted forest:**
Returns a forest or list of forests selected at random.

Returns one or more (depending on `number_of_forests`)
unlabeled rooted forests with `n` nodes, and with no more than
`q` nodes per tree, drawn uniformly at random.
The "roots" graph attribute identifies the roots of the forest.
parameters:
  - n: <class 'int'> = ? --The number of nodes
  - q: int | None = ? --The maximum number of nodes per tree.
  - number_of_forests: int | None = ? --If not None, this number of forests is generated and returned.
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.generators.trees.random_unlabeled_rooted_forest(n=<n_value>, q=<q_value>, number_of_forests=<number_of_forests_value>, seed=<seed_value>)
