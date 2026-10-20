**Directed configuration model:**
Returns a directed_random graph with the given degree sequences.

The configuration model generates a random directed pseudograph
(graph with parallel edges and self loops) by randomly assigning
edges to match the given degree sequences.
parameters:
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.generators.degree_seq.directed_configuration_model(seed=<seed_value>)
