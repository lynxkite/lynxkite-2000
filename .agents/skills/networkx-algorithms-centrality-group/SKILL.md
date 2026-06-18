---
name: networkx-algorithms-centrality-group
description: Collection of operations - Group betweenness centrality, Prominent group
---

**Group betweenness centrality:**
Compute the group betweenness centrality for a group of nodes.

Group betweenness centrality of a group of nodes $C$ is the sum of the
fraction of all-pairs shortest paths that pass through any vertex in $C$

.. math::

   c_B(v) =\sum_{s,t \in V} \frac{\sigma(s, t|v)}{\sigma(s, t)}

where $V$ is the set of nodes, $\sigma(s, t)$ is the number of
shortest $(s, t)$-paths, and $\sigma(s, t|C)$ is the number of
those paths passing through some node in group $C$. Note that
$(s, t)$ are not members of the group ($V-C$ is the set of nodes
in $V$ that are not in $C$).
parameters:
  - normalized: bool | None = None - .
  - weight: str | None = None - .
  - endpoints: bool | None = None - .
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.algorithms.centrality.group.group_betweenness_centrality(normalized=<normalized_value>, weight=<weight_value>, endpoints=<endpoints_value>, G=<G_variable>)

**Prominent group:**
Find the prominent group of size $k$ in graph $G$. The prominence of the
group is evaluated by the group betweenness centrality.

Group betweenness centrality of a group of nodes $C$ is the sum of the
fraction of all-pairs shortest paths that pass through any vertex in $C$

.. math::

   c_B(v) =\sum_{s,t \in V} \frac{\sigma(s, t|v)}{\sigma(s, t)}

where $V$ is the set of nodes, $\sigma(s, t)$ is the number of
shortest $(s, t)$-paths, and $\sigma(s, t|C)$ is the number of
those paths passing through some node in group $C$. Note that
$(s, t)$ are not members of the group ($V-C$ is the set of nodes
in $V$ that are not in $C$).
parameters:
  - k: <class 'int'> = None - .
  - weight: str | None = None - .
  - endpoints: bool | None = None - .
  - normalized: bool | None = None - .
  - greedy: bool | None = None - .
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.algorithms.centrality.group.prominent_group(k=<k_value>, weight=<weight_value>, endpoints=<endpoints_value>, normalized=<normalized_value>, greedy=<greedy_value>, G=<G_variable>)
