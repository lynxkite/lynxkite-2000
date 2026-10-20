**Random Internet as graph:**
Generates a random undirected graph resembling the Internet AS network
parameters:
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.generators.internet_as_graphs.random_internet_as_graph(seed=<seed_value>)
