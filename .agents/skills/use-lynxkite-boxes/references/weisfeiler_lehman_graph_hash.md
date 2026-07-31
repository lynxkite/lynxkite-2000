**Weisfeiler–Lehman graph hash:**
Return Weisfeiler Lehman (WL) graph hash.

.. Warning:: Hash values for directed graphs and graphs without edge or
    node attributes have changed in v3.5. In previous versions,
    directed graphs did not distinguish in- and outgoing edges. Also,
    graphs without attributes set initial states such that effectively
    one extra iteration of WL occurred than indicated by `iterations`.
    For undirected graphs without node or edge labels, the old
    hashes can be obtained by increasing the iteration count by one.
    For more details, see `issue #7806
    <https://github.com/networkx/networkx/issues/7806>`_.

The function iteratively aggregates and hashes neighborhoods of each node.
After each node's neighbors are hashed to obtain updated node labels,
a hashed histogram of resulting labels is returned as the final hash.

Hashes are identical for isomorphic graphs and strong guarantees that
non-isomorphic graphs will get different hashes. See [1]_ for details.

If no node or edge attributes are provided, the degree of each node
is used as its initial label.
Otherwise, node and/or edge labels are used to compute the hash.
parameters:
  - edge_attr: str | None = ? --The key in edge attribute dictionary to be used for hashing.
If None, edge labels are ignored.
  - node_attr: str | None = ? --The key in node attribute dictionary to be used for hashing.
If None, and no edge_attr given, use the degrees of the nodes as labels.
  - iterations: int | None = 3 --Number of neighbor aggregations to perform.
Should be larger for larger graphs.
  - digest_size: int | None = 16 --Size (in bytes) of blake2b hash digest to use for hashing node labels.
  - G: <class 'networkx.classes.graph.Graph'> = ? --The graph to be hashed.
Can have node and/or edge attributes. Can also have no attributes.
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash(edge_attr=<edge_attr_value>, node_attr=<node_attr_value>, iterations=<iterations_value>, digest_size=<digest_size_value>, G=<G_variable>)
