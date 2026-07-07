**G(n,c) graph:**
Returns the growing network with copying (GNC) digraph with `n` nodes.

The GNC graph is built by adding nodes one at a time with a link to one
previously added node (chosen uniformly at random) and to all of that
node's successors.
parameters:
  - n: <class 'int'> = ? --The number of nodes for the generated graph.
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.generators.directed.gnc_graph(n=<n_value>, seed=<seed_value>)
