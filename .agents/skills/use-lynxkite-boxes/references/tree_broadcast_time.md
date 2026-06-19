**Tree broadcast time:**
Return the minimum broadcast time of a (node in a) tree.

The minimum broadcast time of a node is defined as the minimum amount
of time required to complete broadcasting starting from that node.
The broadcast time of a graph is the maximum over
all nodes of the minimum broadcast time from that node [1]_.
This function returns the minimum broadcast time of `node`.
If `node` is `None`, the broadcast time for the graph is returned.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --The graph should be an undirected tree.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.broadcasting.tree_broadcast_time(G=<G_variable>)
