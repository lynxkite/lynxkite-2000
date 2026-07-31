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
