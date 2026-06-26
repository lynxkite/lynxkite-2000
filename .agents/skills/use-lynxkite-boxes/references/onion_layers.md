**Onion layers:**
Returns the layer of each vertex in an onion decomposition of the graph.

The onion decomposition refines the k-core decomposition by providing
information on the internal organization of each k-shell. It is usually
used alongside the `core numbers`.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --An undirected graph without self loops.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.core.onion_layers(G=<G_variable>)
