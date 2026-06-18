---
name: networkx-convert-matrix
description: Collection of operations - From Pandas adjacency, From Pandas edgelist, To Pandas edgelist, From SciPy sparse array, To SciPy sparse array, From NumPy array, To NumPy array
---

**From Pandas adjacency:**
Returns a graph from Pandas DataFrame.

The Pandas DataFrame is interpreted as an adjacency matrix for the graph.
parameters:


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


usage:
output_variable = networkx.convert_matrix.from_pandas_edgelist()

**To Pandas edgelist:**
Returns the graph edge list as a Pandas DataFrame.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.convert_matrix.to_pandas_edgelist(G=<G_variable>)

**From SciPy sparse array:**
Creates a new graph from an adjacency matrix given as a SciPy sparse
array.
parameters:
  - parallel_edges: <class 'bool'> = None - .
  - edge_attribute: <class 'str'> = weight - .

usage:
output_variable = networkx.convert_matrix.from_scipy_sparse_array(parallel_edges=<parallel_edges_value>, edge_attribute=<edge_attribute_value>)

**To SciPy sparse array:**
Returns the graph adjacency matrix as a SciPy sparse array.
parameters:
  - weight: str | None = weight - .
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.convert_matrix.to_scipy_sparse_array(weight=<weight_value>, G=<G_variable>)

**From NumPy array:**
Returns a graph from a 2D NumPy array.

The 2D NumPy array is interpreted as an adjacency matrix for the graph.
parameters:
  - parallel_edges: <class 'bool'> = None - .
  - edge_attr: str | None = weight - .

usage:
output_variable = networkx.convert_matrix.from_numpy_array(parallel_edges=<parallel_edges_value>, edge_attr=<edge_attr_value>)

**To NumPy array:**
Returns the graph adjacency matrix as a NumPy array.
parameters:
  - weight: <class 'str'> = weight - .
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.convert_matrix.to_numpy_array(weight=<weight_value>, G=<G_variable>)
