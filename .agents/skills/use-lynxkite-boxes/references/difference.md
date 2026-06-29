**Difference:**
Returns a new graph that contains the edges that exist in G but not in H.

The node sets of H and G must be the same.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --A NetworkX graph. G and H must have the same node sets.
  - H: <class 'networkx.classes.graph.Graph'> = ? --?

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.operators.binary.difference(G=<G_variable>, H=<H_variable>)
