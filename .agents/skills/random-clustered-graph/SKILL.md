---
name: random-clustered-graph
description: Generate a random graph with the given joint independent edge degree and
---

**Random clustered graph:**
Generate a random graph with the given joint independent edge degree and
triangle degree sequence.

This uses a configuration model-like approach to generate a random graph
(with parallel edges and self-loops) by randomly assigning edges to match
the given joint degree sequence.

The joint degree sequence is a list of pairs of integers of the form
$[(d_{1,i}, d_{1,t}), \dotsc, (d_{n,i}, d_{n,t})]$. According to this list,
vertex $u$ is a member of $d_{u,t}$ triangles and has $d_{u, i}$ other
edges. The number $d_{u,t}$ is the *triangle degree* of $u$ and the number
$d_{u,i}$ is the *independent edge degree*.
parameters:
  - seed: int | None = None -

usage:
output_variable = networkx.generators.random_clustered.random_clustered_graph(seed=<seed_value>)
