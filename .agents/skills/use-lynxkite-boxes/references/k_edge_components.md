**K edge components:**
Generates nodes in each maximal k-edge-connected component in G.
parameters:
  - k: <class 'int'> = ? --Desired edge connectivity
  - G: <class 'networkx.classes.graph.Graph'> = ? --?
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.connectivity.edge_kcomponents.k_edge_components(k=<k_value>, G=<G_variable>)
