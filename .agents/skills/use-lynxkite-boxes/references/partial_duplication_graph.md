**Partial duplication graph:**
Returns a random graph using the partial duplication model.
parameters:
  - N: <class 'int'> = ? --The total number of nodes in the final graph.
  - n: <class 'int'> = ? --The number of nodes in the initial clique.
  - p: <class 'float'> = ? --The probability of joining each neighbor of a node to the
duplicate node. Must be a number in the between zero and one,
inclusive.
  - q: <class 'float'> = ? --The probability of joining the source node to the duplicate
node. Must be a number in the between zero and one, inclusive.
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.generators.duplication.partial_duplication_graph(N=<N_value>, n=<n_value>, p=<p_value>, q=<q_value>, seed=<seed_value>)
