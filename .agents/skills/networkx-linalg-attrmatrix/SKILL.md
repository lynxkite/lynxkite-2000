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
  - edge_attr: str | None = None - .
  - node_attr: str | None = None - .
  - normalized: bool | None = None - .
  - G: <class 'networkx.classes.graph.Graph'> = None - .

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
  - edge_attr: str | None = None - .
  - node_attr: str | None = None - .
  - normalized: bool | None = None - .
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.linalg.attrmatrix.attr_sparse_matrix(edge_attr=<edge_attr_value>, node_attr=<node_attr_value>, normalized=<normalized_value>, G=<G_variable>)
