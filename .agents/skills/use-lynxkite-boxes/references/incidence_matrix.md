**Incidence matrix:**
Returns incidence matrix of G.

The incidence matrix assigns each row to a node and each column to an edge.
For a standard incidence matrix a 1 appears wherever a row's node is
incident on the column's edge.  For an oriented incidence matrix each
edge is assigned an orientation (arbitrarily for undirected and aligning to
direction for directed).  A -1 appears for the source (tail) of an edge and
1 for the destination (head) of the edge.  The elements are zero otherwise.
parameters:
  - oriented: bool | None = ? --If True, matrix elements are +1 or -1 for the head or tail node
respectively of each edge.  If False, +1 occurs at both nodes.
  - weight: str | None = ? --The edge data key used to provide each value in the matrix.
If None, then each edge has weight 1.  Edge weights, if used,
should be positive so that the orientation can provide the sign.
  - G: <class 'networkx.classes.graph.Graph'> = ? --A NetworkX graph

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.linalg.graphmatrix.incidence_matrix(oriented=<oriented_value>, weight=<weight_value>, G=<G_variable>)
