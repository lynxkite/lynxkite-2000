**KL connected subgraph:**
Returns the maximum locally `(k, l)`-connected subgraph of `G`.

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
  - same_as_graph: <class 'bool'> = ? --If True then return a tuple of the form `(H, is_same)`,
where `H` is the maximum locally `(k, l)`-connected subgraph and
`is_same` is a Boolean representing whether `G` is locally `(k,
l)`-connected (and hence, whether `H` is simply a copy of the input
graph `G`).
  - G: <class 'networkx.classes.graph.Graph'> = ? --The graph in which to find a maximum locally `(k, l)`-connected
subgraph.
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.hybrid.kl_connected_subgraph(k=<k_value>, l=<l_value>, low_memory=<low_memory_value>, same_as_graph=<same_as_graph_value>, G=<G_variable>)
