**Eulerian path:**
Return an iterator over the edges of an Eulerian path in `G`.
parameters:
  - keys: <class 'bool'> = ? --Indicates whether to yield edge 3-tuples (u, v, edge_key).
The default yields edge 2-tuples
  - G: <class 'networkx.classes.graph.Graph'> = ? --The graph in which to look for an eulerian path.
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.euler.eulerian_path(keys=<keys_value>, G=<G_variable>)
