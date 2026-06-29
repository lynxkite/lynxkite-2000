**From NumPy array:**
Returns a graph from a 2D NumPy array.

The 2D NumPy array is interpreted as an adjacency matrix for the graph.
parameters:
  - parallel_edges: <class 'bool'> = ? --If this is True, `create_using` is a multigraph, and `A` is an
integer array, then entry *(i, j)* in the array is interpreted as the
number of parallel edges joining vertices *i* and *j* in the graph.
If it is False, then the entries in the array are interpreted as
the weight of a single edge joining the vertices.
  - edge_attr: str | None = weight --The attribute to which the array values are assigned on each edge. If
it is None, edge attributes will not be assigned.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.convert_matrix.from_numpy_array(parallel_edges=<parallel_edges_value>, edge_attr=<edge_attr_value>)
