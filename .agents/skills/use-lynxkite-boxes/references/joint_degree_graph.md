**Joint degree graph:**
Generates a random simple graph with the given joint degree dictionary.
parameters:
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.generators.joint_degree_seq.joint_degree_graph(seed=<seed_value>)
