**Random regular graph:**
Returns a random $d$-regular graph on $n$ nodes.

A regular graph is a graph where each node has the same number of neighbors.

The resulting graph has no self-loops or parallel edges.
parameters:
  - d: <class 'int'> = ? --The degree of each node.
  - n: <class 'int'> = ? --The number of nodes. The value of $n \times d$ must be even.
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.random_graphs.random_regular_graph(d=<d_value>, n=<n_value>, seed=<seed_value>)
