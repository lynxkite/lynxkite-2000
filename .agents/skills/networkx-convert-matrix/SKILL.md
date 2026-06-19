---
name: networkx-convert-matrix
description: Collection of operations - From Pandas adjacency, From Pandas edgelist, To Pandas edgelist, From SciPy sparse array, To SciPy sparse array, From NumPy array, To NumPy array
---

**From Pandas adjacency:**
Returns a graph from Pandas DataFrame.

The Pandas DataFrame is interpreted as an adjacency matrix for the graph.
parameters:


returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.convert_matrix.from_pandas_adjacency()

**From Pandas edgelist:**
Returns a graph from Pandas DataFrame containing an edge list.

The Pandas DataFrame should contain at least two columns of node names and
zero or more columns of edge attributes. Each row will be processed as one
edge instance.

Note: This function iterates over DataFrame.values, which is not
guaranteed to retain the data type across columns in the row. This is only
a problem if your row is entirely numeric and a mix of ints and floats. In
that case, all values will be returned as floats. See the
DataFrame.iterrows documentation for an example.
parameters:


returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.convert_matrix.from_pandas_edgelist()

**To Pandas edgelist:**
Returns the graph edge list as a Pandas DataFrame.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --The NetworkX graph used to construct the Pandas DataFrame.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.convert_matrix.to_pandas_edgelist(G=<G_variable>)

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

**To SciPy sparse array:**
Returns the graph adjacency matrix as a SciPy sparse array.
parameters:
  - weight: str | None = weight --The edge attribute that holds the numerical value used for
the edge weight.  If None then all edge weights are 1.
  - G: <class 'networkx.classes.graph.Graph'> = ? --The NetworkX graph used to construct the sparse array.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.convert_matrix.to_scipy_sparse_array(weight=<weight_value>, G=<G_variable>)

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

**To NumPy array:**
Returns the graph adjacency matrix as a NumPy array.
parameters:
  - weight: <class 'str'> = weight --The edge attribute that holds the numerical value used for
the edge weight. If an edge does not have that attribute, then the
value 1 is used instead. `weight` must be ``None`` if a structured
dtype is used.
  - G: <class 'networkx.classes.graph.Graph'> = ? --The NetworkX graph used to construct the NumPy array.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.convert_matrix.to_numpy_array(weight=<weight_value>, G=<G_variable>)
