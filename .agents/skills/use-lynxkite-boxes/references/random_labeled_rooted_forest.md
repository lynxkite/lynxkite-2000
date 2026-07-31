**Random labeled rooted forest:**
Returns a labeled rooted forest with `n` nodes.

The returned forest is chosen uniformly at random using a
generalization of Prüfer sequences [1]_ in the form described in [2]_.
parameters:
  - n: <class 'int'> = ? --The number of nodes.
  - seed: int | None = ? --See :ref:`Randomness<randomness>`.
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.generators.trees.random_labeled_rooted_forest(n=<n_value>, seed=<seed_value>)
