**Cn Soundarajan–Hopcroft:**
Count the number of common neighbors of all node pairs in ebunch
    using community information.

For two nodes $u$ and $v$, this function computes the number of
common neighbors and bonus one for each common neighbor belonging to
the same community as $u$ and $v$. Mathematically,

.. math::

    |\Gamma(u) \cap \Gamma(v)| + \sum_{w \in \Gamma(u) \cap \Gamma(v)} f(w)

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
  output_variable = networkx.algorithms.link_prediction.cn_soundarajan_hopcroft(community=<community_value>, G=<G_variable>)
