**From SciPy sparse array:**
Creates a new graph from an adjacency matrix given as a SciPy sparse
array.
parameters:
  - parallel_edges: <class 'bool'> = ? --If this is True, `create_using` is a multigraph, and `A` is an
integer matrix, then entry *(i, j)* in the matrix is interpreted as the
number of parallel edges joining vertices *i* and *j* in the graph.
If it is False, then the entries in the matrix are interpreted as
the weight of a single edge joining the vertices.
  - edge_attribute: <class 'str'> = weight --Name of edge attribute to store matrix numeric value. The data will
have the same type as the matrix entry (int, float, (real,imag)).

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.convert_matrix.from_scipy_sparse_array(parallel_edges=<parallel_edges_value>, edge_attribute=<edge_attribute_value>)
