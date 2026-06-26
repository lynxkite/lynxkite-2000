**Ra index Soundarajan–Hopcroft:**
Compute the resource allocation index of all node pairs in
ebunch using community information.

For two nodes $u$ and $v$, this function computes the resource
allocation index considering only common neighbors belonging to the
same community as $u$ and $v$. Mathematically,

.. math::

    \sum_{w \in \Gamma(u) \cap \Gamma(v)} \frac{f(w)}{|\Gamma(w)|}

where $f(w)$ equals 1 if $w$ belongs to the same community as $u$
and $v$ or 0 otherwise and $\Gamma(u)$ denotes the set of
neighbors of $u$.
parameters:
  - community: str | None = community --Nodes attribute name containing the community information.
G[u][community] identifies which community u belongs to. Each
node belongs to at most one community. Default value: 'community'.
  - G: <class 'networkx.classes.graph.Graph'> = ? --A NetworkX undirected graph.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.link_prediction.ra_index_soundarajan_hopcroft(community=<community_value>, G=<G_variable>)
