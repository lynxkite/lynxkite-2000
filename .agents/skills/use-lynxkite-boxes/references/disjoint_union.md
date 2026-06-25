**Disjoint union:**
Combine graphs G and H. The nodes are assumed to be unique (disjoint).

This algorithm automatically relabels nodes to avoid name collisions.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --A NetworkX graph
  - H: <class 'networkx.classes.graph.Graph'> = ? --?
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.operators.binary.disjoint_union(G=<G_variable>, H=<H_variable>)
