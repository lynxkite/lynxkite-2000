---
name: spectral-graph-forge
description: Returns a random simple graph with spectrum resembling that of `G`
---

**Spectral graph forge:**
Returns a random simple graph with spectrum resembling that of `G`

This algorithm, called Spectral Graph Forge (SGF), computes the
eigenvectors of a given graph adjacency matrix, filters them and
builds a random graph with a similar eigenstructure.
SGF has been proved to be particularly useful for synthesizing
realistic social networks and it can also be used to anonymize
graph sensitive data.
parameters:
  - alpha: <class 'float'> = ? --Ratio representing the percentage of eigenvectors of G to consider,
values in [0,1].
  - transformation: str | None = identity --Represents the intended matrix linear transformation, possible values
are 'identity' and 'modularity'
  - seed: int | None = ? --Indicator of numpy random number generation state.
See :ref:`Randomness<randomness>`.
  - G: <class 'networkx.classes.graph.Graph'> = ? --?

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.spectral_graph_forge.spectral_graph_forge(alpha=<alpha_value>, transformation=<transformation_value>, seed=<seed_value>, G=<G_variable>)
