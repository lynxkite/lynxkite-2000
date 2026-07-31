**Union:**
Combine graphs G and H. The names of nodes must be unique.

A name collision between the graphs will raise an exception.

A renaming facility is provided to avoid name collisions.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --A NetworkX graph
  - H: <class 'networkx.classes.graph.Graph'> = ? --A NetworkX graph
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.operators.binary.union(G=<G_variable>, H=<H_variable>)
