---
name: networkx-algorithms-centrality-betweenness
description: Collection of operations - Betweenness centrality, Edge betweenness centrality
---

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
  - k: int | None = None -
  - normalized: bool | None = None -
  - weight: str | None = None -
  - endpoints: bool | None = None -
  - seed: int | None = None -
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.centrality.betweenness.betweenness_centrality(k=<k_value>, normalized=<normalized_value>, weight=<weight_value>, endpoints=<endpoints_value>, seed=<seed_value>, G=<G_variable>)

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
  - k: int | None = None -
  - normalized: bool | None = None -
  - weight: str | None = None -
  - seed: int | None = None -
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.centrality.betweenness.edge_betweenness_centrality(k=<k_value>, normalized=<normalized_value>, weight=<weight_value>, seed=<seed_value>, G=<G_variable>)
