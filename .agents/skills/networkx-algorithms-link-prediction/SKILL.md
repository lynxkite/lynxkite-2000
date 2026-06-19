---
name: networkx-algorithms-link-prediction
description: Collection of operations - Resource allocation index, Jaccard coefficient, Adamic–Adar index, Preferential attachment, Cn Soundarajan–Hopcroft, Ra index Soundarajan–Hopcroft, Within inter cluster, Common neighbor centrality
---

**Resource allocation index:**
Compute the resource allocation index of all node pairs in ebunch.

Resource allocation index of `u` and `v` is defined as

.. math::

    \sum_{w \in \Gamma(u) \cap \Gamma(v)} \frac{1}{|\Gamma(w)|}

where $\Gamma(u)$ denotes the set of neighbors of $u$.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --A NetworkX undirected graph.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.link_prediction.resource_allocation_index(G=<G_variable>)

**Jaccard coefficient:**
Compute the Jaccard coefficient of all node pairs in ebunch.

Jaccard coefficient of nodes `u` and `v` is defined as

.. math::

    \frac{|\Gamma(u) \cap \Gamma(v)|}{|\Gamma(u) \cup \Gamma(v)|}

where $\Gamma(u)$ denotes the set of neighbors of $u$.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --A NetworkX undirected graph.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.link_prediction.jaccard_coefficient(G=<G_variable>)

**Adamic–Adar index:**
Compute the Adamic-Adar index of all node pairs in ebunch.

Adamic-Adar index of `u` and `v` is defined as

.. math::

    \sum_{w \in \Gamma(u) \cap \Gamma(v)} \frac{1}{\log |\Gamma(w)|}

where $\Gamma(u)$ denotes the set of neighbors of $u$.
This index leads to zero-division for nodes only connected via self-loops.
It is intended to be used when no self-loops are present.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --NetworkX undirected graph.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.link_prediction.adamic_adar_index(G=<G_variable>)

**Preferential attachment:**
Compute the preferential attachment score of all node pairs in ebunch.

Preferential attachment score of `u` and `v` is defined as

.. math::

    |\Gamma(u)| |\Gamma(v)|

where $\Gamma(u)$ denotes the set of neighbors of $u$.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --NetworkX undirected graph.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.link_prediction.preferential_attachment(G=<G_variable>)

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

**Within inter cluster:**
Compute the ratio of within- and inter-cluster common neighbors
of all node pairs in ebunch.

For two nodes `u` and `v`, if a common neighbor `w` belongs to the
same community as them, `w` is considered as within-cluster common
neighbor of `u` and `v`. Otherwise, it is considered as
inter-cluster common neighbor of `u` and `v`. The ratio between the
size of the set of within- and inter-cluster common neighbors is
defined as the WIC measure. [1]_
parameters:
  - delta: float | None = 0.001 --Value to prevent division by zero in case there is no
inter-cluster common neighbor between two nodes. See [1]_ for
details. Default value: 0.001.
  - community: str | None = community --Nodes attribute name containing the community information.
G[u][community] identifies which community u belongs to. Each
node belongs to at most one community. Default value: 'community'.
  - G: <class 'networkx.classes.graph.Graph'> = ? --A NetworkX undirected graph.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.link_prediction.within_inter_cluster(delta=<delta_value>, community=<community_value>, G=<G_variable>)

**Common neighbor centrality:**
Return the CCPA score for each pair of nodes.

Compute the Common Neighbor and Centrality based Parameterized Algorithm(CCPA)
score of all node pairs in ebunch.

CCPA score of `u` and `v` is defined as

.. math::

    \alpha \cdot (|\Gamma (u){\cap }^{}\Gamma (v)|)+(1-\alpha )\cdot \frac{N}{{d}_{uv}}

where $\Gamma(u)$ denotes the set of neighbors of $u$, $\Gamma(v)$ denotes the
set of neighbors of $v$, $\alpha$ is  parameter varies between [0,1], $N$ denotes
total number of nodes in the Graph and ${d}_{uv}$ denotes shortest distance
between $u$ and $v$.

This algorithm is based on two vital properties of nodes, namely the number
of common neighbors and their centrality. Common neighbor refers to the common
nodes between two nodes. Centrality refers to the prestige that a node enjoys
in a network.

.. seealso::

    :func:`common_neighbors`
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --NetworkX undirected graph.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.link_prediction.common_neighbor_centrality(G=<G_variable>)
