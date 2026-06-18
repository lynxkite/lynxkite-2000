---
name: generate-random-paths
description: Randomly generate `sample_size` paths of length `path_length`.
---

**Generate random paths:**
Randomly generate `sample_size` paths of length `path_length`.
parameters:
  - sample_size: <class 'int'> = None -
  - path_length: <class 'int'> = 5 -
  - weight: str | None = weight -
  - seed: int | None = None -
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.similarity.generate_random_paths(sample_size=<sample_size_value>, path_length=<path_length_value>, weight=<weight_value>, seed=<seed_value>, G=<G_variable>)
