**Tree broadcast center:**
Return the broadcast center of a tree.

The broadcast center of a graph `G` denotes the set of nodes having
minimum broadcast time [1]_. This function implements a linear algorithm
for determining the broadcast center of a tree with ``n`` nodes. As a
by-product, it also determines the broadcast time from the broadcast center.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --The graph should be an undirected tree.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.broadcasting.tree_broadcast_center(G=<G_variable>)
