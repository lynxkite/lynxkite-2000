**Maybe regular expander graph:**
Utility for creating a random regular expander.

Returns a random $d$-regular graph on $n$ nodes which is an expander
graph with very good probability.
parameters:
  - n: <class 'int'> = ? --The number of nodes.
  - d: <class 'int'> = ? --The degree of each node.
  - max_tries: <class 'int'> = 100 --The number of allowed loops when generating each independent cycle
  - seed: int | None = ? --Seed used to set random number generation state. See :ref`Randomness<randomness>`.
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.generators.expanders.maybe_regular_expander_graph(n=<n_value>, d=<d_value>, max_tries=<max_tries_value>, seed=<seed_value>)
