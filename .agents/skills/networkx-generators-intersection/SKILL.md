---
name: networkx-generators-intersection
description: Collection of operations - Uniform random intersection graph, K random intersection graph, General random intersection graph
---

**Uniform random intersection graph:**
Returns a uniform random intersection graph.
parameters:
  - n: <class 'int'> = None -
  - m: <class 'int'> = None -
  - p: <class 'float'> = None -
  - seed: int | None = None -

usage:
output_variable = networkx.generators.intersection.uniform_random_intersection_graph(n=<n_value>, m=<m_value>, p=<p_value>, seed=<seed_value>)

**K random intersection graph:**
Returns a intersection graph with randomly chosen attribute sets for
each node that are of equal size (k).
parameters:
  - n: <class 'int'> = None -
  - m: <class 'int'> = None -
  - k: <class 'float'> = None -
  - seed: int | None = None -

usage:
output_variable = networkx.generators.intersection.k_random_intersection_graph(n=<n_value>, m=<m_value>, k=<k_value>, seed=<seed_value>)

**General random intersection graph:**
Returns a random intersection graph with independent probabilities
for connections between node and attribute sets.
parameters:
  - n: <class 'int'> = None -
  - m: <class 'int'> = None -
  - seed: int | None = None -

usage:
output_variable = networkx.generators.intersection.general_random_intersection_graph(n=<n_value>, m=<m_value>, seed=<seed_value>)
