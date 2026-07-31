**Resistance distance:**
Returns the resistance distance between pairs of nodes in graph G.

The resistance distance between two nodes of a graph is akin to treating
the graph as a grid of resistors with a resistance equal to the provided
weight [1]_, [2]_.

If weight is not provided, then a weight of 1 is used for all edges.

If two nodes are the same, the resistance distance is zero.
parameters:
  - weight: str | None = ? --The edge data key used to compute the resistance distance.
If None, then each edge has weight 1.
  - invert_weight: <class 'bool'> = ? --Proper calculation of resistance distance requires building the
Laplacian matrix with the reciprocal of the weight. Not required
if the weight is already inverted. Weight cannot be zero.
  - G: <class 'networkx.classes.graph.Graph'> = ? --A graph
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.distance_measures.resistance_distance(weight=<weight_value>, invert_weight=<invert_weight_value>, G=<G_variable>)
