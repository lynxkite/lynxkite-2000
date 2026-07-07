**Margulis–Gabber–Galil graph:**
Returns the Margulis-Gabber-Galil undirected MultiGraph on `n^2` nodes.

The undirected MultiGraph is regular with degree `8`. Nodes are integer
pairs. The second-largest eigenvalue of the adjacency matrix of the graph
is at most `5 \sqrt{2}`, regardless of `n`.
parameters:
  - n: <class 'int'> = ? --Determines the number of nodes in the graph: `n^2`.
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.generators.expanders.margulis_gabber_galil_graph(n=<n_value>)
