**Configuration model:**
Returns a random graph with the given degree sequence.

The configuration model generates a random pseudograph (graph with
parallel edges and self loops) by randomly assigning edges to
match the given degree sequence.
parameters:
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.generators.degree_seq.configuration_model(seed=<seed_value>)
