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
  - k: <class 'int'> = ? --The number of nodes in the group.
  - weight: str | None = ? --If None, all edge weights are considered equal.
Otherwise holds the name of the edge attribute used as weight.
The weight of an edge is treated as the length or distance between the two sides.
  - endpoints: bool | None = ? --If True include the endpoints in the shortest path counts.
  - normalized: bool | None = ? --If True, group betweenness is normalized by ``1/((|V|-|C|)(|V|-|C|-1))``
where ``|V|`` is the number of nodes in G and ``|C|`` is the number of
nodes in C.
  - greedy: bool | None = ? --Using a naive greedy algorithm in order to find non-optimal prominent
group. For scale free networks the results are negligibly below the optimal
results.
  - G: <class 'networkx.classes.graph.Graph'> = ? --A NetworkX graph.
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.centrality.group.prominent_group(k=<k_value>, weight=<weight_value>, endpoints=<endpoints_value>, normalized=<normalized_value>, greedy=<greedy_value>, G=<G_variable>)
