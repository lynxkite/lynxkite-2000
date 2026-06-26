**Generalized degree:**
Compute the generalized degree for nodes.

For each node, the generalized degree shows how many edges of given
triangle multiplicity the node is connected to. The triangle multiplicity
of an edge is the number of triangles an edge participates in. The
generalized degree of node :math:`i` can be written as a vector
:math:`\mathbf{k}_i=(k_i^{(0)}, \dotsc, k_i^{(N-2)})` where
:math:`k_i^{(j)}` is the number of edges attached to node :math:`i` that
participate in :math:`j` triangles.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --?

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.cluster.generalized_degree(G=<G_variable>)
