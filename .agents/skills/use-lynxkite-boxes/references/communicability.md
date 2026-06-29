**Communicability:**
Returns communicability between all pairs of nodes in G.

The communicability between pairs of nodes in G is the sum of
walks of different lengths starting at node u and ending at node v.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --?

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.communicability_alg.communicability(G=<G_variable>)
