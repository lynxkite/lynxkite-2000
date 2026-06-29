**Is KL connected:**
Returns True if and only if `G` is locally `(k, l)`-connected.

A graph is locally `(k, l)`-connected if for each edge `(u, v)` in the
graph there are at least `l` edge-disjoint paths of length at most `k`
joining `u` to `v`.
parameters:
  - k: <class 'int'> = ? --The maximum length of paths to consider. A higher number means a looser
connectivity requirement.
  - l: <class 'int'> = ? --The number of edge-disjoint paths. A higher number means a stricter
connectivity requirement.
  - low_memory: <class 'bool'> = ? --If this is True, this function uses an algorithm that uses slightly
more time but less memory.
  - G: <class 'networkx.classes.graph.Graph'> = ? --The graph to test for local `(k, l)`-connectedness.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.hybrid.is_kl_connected(k=<k_value>, l=<l_value>, low_memory=<low_memory_value>, G=<G_variable>)
