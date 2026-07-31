**Generate random paths:**
Randomly generate `sample_size` paths of length `path_length`.
parameters:
  - sample_size: <class 'int'> = ? --The number of paths to generate. This is ``R`` in [1]_.
  - path_length: <class 'int'> = 5 --The maximum size of the path to randomly generate.
This is ``T`` in [1]_. According to the paper, ``T >= 5`` is
recommended.
  - weight: str | None = weight --The name of an edge attribute that holds the numerical value
used as a weight. If None then each edge has weight 1.
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.
  - G: <class 'networkx.classes.graph.Graph'> = ? --A NetworkX graph
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.similarity.generate_random_paths(sample_size=<sample_size_value>, path_length=<path_length_value>, weight=<weight_value>, seed=<seed_value>, G=<G_variable>)
