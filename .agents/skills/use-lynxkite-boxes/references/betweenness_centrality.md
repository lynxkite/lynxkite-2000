**Betweenness centrality:**
Compute the shortest-path betweenness centrality for nodes.

Betweenness centrality of a node $v$ is the sum of the
fraction of all-pairs shortest paths that pass through $v$.

.. math::

   c_B(v) = \sum_{s, t \in V} \frac{\sigma(s, t | v)}{\sigma(s, t)}

where $V$ is the set of nodes, $\sigma(s, t)$ is the number of
shortest $(s, t)$-paths, and $\sigma(s, t | v)$ is the number of
those paths passing through some node $v$ other than $s$ and $t$.
If $s = t$, $\sigma(s, t) = 1$, and if $v \in \{s, t\}$,
$\sigma(s, t | v) = 0$ [2]_.
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
  - endpoints: bool | None = ? --If `True`, include the endpoints $s$ and $t$ in the shortest path counts.
This is taken into account when rescaling the values.
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.
Note that this is only used if ``k is not None``.
  - G: <class 'networkx.classes.graph.Graph'> = ? --A NetworkX graph.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.centrality.betweenness.betweenness_centrality(k=<k_value>, normalized=<normalized_value>, weight=<weight_value>, endpoints=<endpoints_value>, seed=<seed_value>, G=<G_variable>)
