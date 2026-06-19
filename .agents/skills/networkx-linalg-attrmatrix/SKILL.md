---
name: networkx-linalg-attrmatrix
description: Collection of operations - Attr matrix, Attr sparse matrix
---

**Attr matrix:**
Returns the attribute matrix using attributes from `G` as a numpy array.

If only `G` is passed in, then the adjacency matrix is constructed.

Let A be a discrete set of values for the node attribute `node_attr`. Then
the elements of A represent the rows and columns of the constructed matrix.
Now, iterate through every edge e=(u,v) in `G` and consider the value
of the edge attribute `edge_attr`.  If ua and va are the values of the
node attribute `node_attr` for u and v, respectively, then the value of
the edge attribute is added to the matrix element at (ua, va).
parameters:
  - edge_attr: str | None = ? --Each element of the matrix represents a running total of the
specified edge attribute for edges whose node attributes correspond
to the rows/cols of the matrix. The attribute must be present for
all edges in the graph. If no attribute is specified, then we
just count the number of edges whose node attributes correspond
to the matrix element.
  - node_attr: str | None = ? --Each row and column in the matrix represents a particular value
of the node attribute.  The attribute must be present for all nodes
in the graph. Note, the values of this attribute should be reliably
hashable. So, float values are not recommended. If no attribute is
specified, then the rows and columns will be the nodes of the graph.
  - normalized: bool | None = ? --If True, then each row is normalized by the summation of its values.
  - G: <class 'networkx.classes.graph.Graph'> = ? --The NetworkX graph used to construct the attribute matrix.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.linalg.attrmatrix.attr_matrix(edge_attr=<edge_attr_value>, node_attr=<node_attr_value>, normalized=<normalized_value>, G=<G_variable>)

**Attr sparse matrix:**
Returns a SciPy sparse array using attributes from G.

If only `G` is passed in, then the adjacency matrix is constructed.

Let A be a discrete set of values for the node attribute `node_attr`. Then
the elements of A represent the rows and columns of the constructed matrix.
Now, iterate through every edge e=(u,v) in `G` and consider the value
of the edge attribute `edge_attr`.  If ua and va are the values of the
node attribute `node_attr` for u and v, respectively, then the value of
the edge attribute is added to the matrix element at (ua, va).
parameters:
  - edge_attr: str | None = ? --Each element of the matrix represents a running total of the
specified edge attribute for edges whose node attributes correspond
to the rows/cols of the matrix. The attribute must be present for
all edges in the graph. If no attribute is specified, then we
just count the number of edges whose node attributes correspond
to the matrix element.
  - node_attr: str | None = ? --Each row and column in the matrix represents a particular value
of the node attribute.  The attribute must be present for all nodes
in the graph. Note, the values of this attribute should be reliably
hashable. So, float values are not recommended. If no attribute is
specified, then the rows and columns will be the nodes of the graph.
  - normalized: bool | None = ? --If True, then each row is normalized by the summation of its values.
  - G: <class 'networkx.classes.graph.Graph'> = ? --The NetworkX graph used to construct the NumPy matrix.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.linalg.attrmatrix.attr_sparse_matrix(edge_attr=<edge_attr_value>, node_attr=<node_attr_value>, normalized=<normalized_value>, G=<G_variable>)
