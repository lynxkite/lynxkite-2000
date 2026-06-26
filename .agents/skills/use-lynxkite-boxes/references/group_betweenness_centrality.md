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
  - normalized: bool | None = ? --If True, group betweenness is normalized by `1/((|V|-|C|)(|V|-|C|-1))`
where `|V|` is the number of nodes in G and `|C|` is the number of nodes in C.
  - weight: str | None = ? --If None, all edge weights are considered equal.
Otherwise holds the name of the edge attribute used as weight.
The weight of an edge is treated as the length or distance between the two sides.
  - endpoints: bool | None = ? --If True include the endpoints in the shortest path counts.
  - G: <class 'networkx.classes.graph.Graph'> = ? --A NetworkX graph.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.centrality.group.group_betweenness_centrality(normalized=<normalized_value>, weight=<weight_value>, endpoints=<endpoints_value>, G=<G_variable>)
