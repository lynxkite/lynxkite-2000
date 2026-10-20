**Number of isolates:**
Returns the number of isolates in the graph.

An *isolate* is a node with no neighbors (that is, with degree
zero). For directed graphs, this means no in-neighbors and no
out-neighbors.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --?
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.isolate.number_of_isolates(G=<G_variable>)
