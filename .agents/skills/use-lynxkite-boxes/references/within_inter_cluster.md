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
