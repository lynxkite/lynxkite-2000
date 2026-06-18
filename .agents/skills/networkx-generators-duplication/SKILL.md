---
name: networkx-generators-duplication
description: Collection of operations - Partial duplication graph, Duplication divergence graph
---

**Partial duplication graph:**
Returns a random graph using the partial duplication model.
parameters:
  - N: <class 'int'> = None - .
  - n: <class 'int'> = None - .
  - p: <class 'float'> = None - .
  - q: <class 'float'> = None - .
  - seed: int | None = None - .

usage:
output_variable = networkx.generators.duplication.partial_duplication_graph(N=<N_value>, n=<n_value>, p=<p_value>, q=<q_value>, seed=<seed_value>)

**Duplication divergence graph:**
Returns an undirected graph using the duplication-divergence model.

A graph of `n` nodes is created by duplicating the initial nodes
and retaining edges incident to the original nodes with a retention
probability `p`.
parameters:
  - n: <class 'int'> = None - .
  - p: <class 'float'> = None - .
  - seed: int | None = None - .

usage:
output_variable = networkx.generators.duplication.duplication_divergence_graph(n=<n_value>, p=<p_value>, seed=<seed_value>)
