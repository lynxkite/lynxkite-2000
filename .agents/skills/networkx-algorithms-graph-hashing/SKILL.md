---
name: networkx-algorithms-graph-hashing
description: Collection of operations - Weisfeiler–Lehman graph hash, Weisfeiler–Lehman subgraph hashes
---

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
  - edge_attr: str | None = None -
  - node_attr: str | None = None -
  - iterations: int | None = 3 -
  - digest_size: int | None = 16 -
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash(edge_attr=<edge_attr_value>, node_attr=<node_attr_value>, iterations=<iterations_value>, digest_size=<digest_size_value>, G=<G_variable>)

**Weisfeiler–Lehman subgraph hashes:**
Return a dictionary of subgraph hashes by node.

.. Warning:: Hash values for directed graphs have changed in version
    v3.5. In previous versions, directed graphs did not distinguish in-
    and outgoing edges.
    Graphs without attributes previously performed an extra iteration of
    WL at initialisation, which was not visible in the output of this
    function. This hash value is now included in the returned dictionary,
    shifting the other calculated hashes one position to the right. To
    obtain the same last subgraph hash, increase the number of iterations
    by one.
    For more details, see `issue #7806
    <https://github.com/networkx/networkx/issues/7806>`_.

Dictionary keys are nodes in `G`, and values are a list of hashes.
Each hash corresponds to a subgraph rooted at a given node u in `G`.
Lists of subgraph hashes are sorted in increasing order of depth from
their root node, with the hash at index i corresponding to a subgraph
of nodes at most i-hops (i edges) distance from u. Thus, each list will contain
`iterations` elements - a hash for a subgraph at each depth. If
`include_initial_labels` is set to `True`, each list will additionally
have contain a hash of the initial node label (or equivalently a
subgraph of depth 0) prepended, totalling ``iterations + 1`` elements.

The function iteratively aggregates and hashes neighborhoods of each node.
This is achieved for each step by replacing for each node its label from
the previous iteration with its hashed 1-hop neighborhood aggregate.
The new node label is then appended to a list of node labels for each
node.

To aggregate neighborhoods for a node $u$ at each step, all labels of
nodes adjacent to $u$ are concatenated. If the `edge_attr` parameter is set,
labels for each neighboring node are prefixed with the value of this attribute
along the connecting edge from this neighbor to node $u$. The resulting string
is then hashed to compress this information into a fixed digest size.

Thus, at the i-th iteration, nodes within i hops influence any given
hashed node label. We can therefore say that at depth $i$ for node $u$
we have a hash for a subgraph induced by the i-hop neighborhood of $u$.

The output can be used to create general Weisfeiler-Lehman graph kernels,
or generate features for graphs or nodes - for example to generate 'words' in
a graph as seen in the 'graph2vec' algorithm.
See [1]_ & [2]_ respectively for details.

Hashes are identical for isomorphic subgraphs and there exist strong
guarantees that non-isomorphic graphs will get different hashes.
See [1]_ for details.

If no node or edge attributes are provided, the degree of each node
is used as its initial label.
Otherwise, node and/or edge labels are used to compute the hash.
parameters:
  - edge_attr: str | None = None -
  - node_attr: str | None = None -
  - iterations: int | None = 3 -
  - digest_size: int | None = 16 -
  - include_initial_labels: bool | None = None -
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.graph_hashing.weisfeiler_lehman_subgraph_hashes(edge_attr=<edge_attr_value>, node_attr=<node_attr_value>, iterations=<iterations_value>, digest_size=<digest_size_value>, include_initial_labels=<include_initial_labels_value>, G=<G_variable>)
