**Symmetric difference:**
Returns new graph with edges that exist in either G or H but not both.

The node sets of H and G must be the same.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --A NetworkX graph.  G and H must have the same node sets.
  - H: <class 'networkx.classes.graph.Graph'> = ? --?
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.operators.binary.symmetric_difference(G=<G_variable>, H=<H_variable>)
