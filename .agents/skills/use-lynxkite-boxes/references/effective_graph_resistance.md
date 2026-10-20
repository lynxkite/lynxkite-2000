**Effective graph resistance:**
Returns the Effective graph resistance of G.

Also known as the Kirchhoff index.

The effective graph resistance is defined as the sum
of the resistance distance of every node pair in G [1]_.

If weight is not provided, then a weight of 1 is used for all edges.

The effective graph resistance of a disconnected graph is infinite.
parameters:
  - weight: str | None = ? --The edge data key used to compute the effective graph resistance.
If None, then each edge has weight 1.
  - invert_weight: <class 'bool'> = ? --Proper calculation of resistance distance requires building the
Laplacian matrix with the reciprocal of the weight. Not required
if the weight is already inverted. Weight cannot be zero.
  - G: <class 'networkx.classes.graph.Graph'> = ? --A graph
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.distance_measures.effective_graph_resistance(weight=<weight_value>, invert_weight=<invert_weight_value>, G=<G_variable>)
