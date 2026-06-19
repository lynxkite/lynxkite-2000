---
name: spanner
description: Returns a spanner of the given graph with the given stretch.
---

**Spanner:**
Returns a spanner of the given graph with the given stretch.

A spanner of a graph G = (V, E) with stretch t is a subgraph
H = (V, E_S) such that E_S is a subset of E and the distance between
any pair of nodes in H is at most t times the distance between the
nodes in G.
parameters:
  - stretch: <class 'float'> = ? --The stretch of the spanner.
  - weight: <class 'str'> = ? --The edge attribute to use as distance.
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.
  - G: <class 'networkx.classes.graph.Graph'> = ? --An undirected simple graph.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.sparsifiers.spanner(stretch=<stretch_value>, weight=<weight_value>, seed=<seed_value>, G=<G_variable>)
