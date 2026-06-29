**Random lobster graph:**
Returns a random lobster graph.

A lobster is a tree that reduces to a caterpillar when pruning all
leaf nodes. A caterpillar is a tree that reduces to a path graph
when pruning all leaf nodes; setting `p2` to zero produces a caterpillar.

This implementation iterates on the probabilities `p1` and `p2` to add
edges at levels 1 and 2, respectively. Graphs are therefore constructed
iteratively with uniform randomness at each level rather than being selected
uniformly at random from the set of all possible lobsters.
parameters:
  - n: <class 'int'> = ? --The expected number of nodes in the backbone
  - p1: <class 'float'> = ? --Probability of adding an edge to the backbone
  - p2: <class 'float'> = ? --Probability of adding an edge one level beyond backbone
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.random_graphs.random_lobster_graph(n=<n_value>, p1=<p1_value>, p2=<p2_value>, seed=<seed_value>)
