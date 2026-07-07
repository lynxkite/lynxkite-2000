**Edge betweenness centrality:**
Compute betweenness centrality for edges.

Betweenness centrality of an edge $e$ is the sum of the
fraction of all-pairs shortest paths that pass through $e$.

.. math::

   c_B(e) = \sum_{s, t \in V} \frac{\sigma(s, t | e)}{\sigma(s, t)}

where $V$ is the set of nodes, $\sigma(s, t)$ is the number of
shortest $(s, t)$-paths, and $\sigma(s, t | e)$ is the number of
those paths passing through edge $e$ [1]_.
The denominator $\sigma(s, t)$ is a normalization factor that can be
turned off to get the raw path counts.
parameters:
  - k: int | None = ? --If `k` is not `None`, use `k` sampled nodes as sources for the considered paths.
The resulting sampled counts are then inflated to approximate betweenness.
Higher values of `k` give better approximation. Must have ``k <= len(G)``.
  - normalized: bool | None = ? --If `True`, the betweenness values are rescaled by dividing by the number of
possible $(s, t)$-pairs in the graph.
  - weight: str | None = ? --If `None`, all edge weights are 1.
Otherwise holds the name of the edge attribute used as weight.
Weights are used to calculate weighted shortest paths, so they are
interpreted as distances.
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.
Note that this is only used if ``k is not None``.
  - G: <class 'networkx.classes.graph.Graph'> = ? --A NetworkX graph.
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.centrality.betweenness.edge_betweenness_centrality(k=<k_value>, normalized=<normalized_value>, weight=<weight_value>, seed=<seed_value>, G=<G_variable>)
