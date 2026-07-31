**Geometric edges:**
Returns edge list of node pairs within `radius` of each other.
parameters:
  - radius: <class 'float'> = ? --The distance threshold. Edges are included in the edge list if the
distance between the two nodes is less than `radius`.
  - G: <class 'networkx.classes.graph.Graph'> = ? --The graph from which to generate the edge list. The nodes in `G` should
have an attribute ``pos`` corresponding to the node position, which is
used to compute the distance to other nodes.
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.generators.geometric.geometric_edges(radius=<radius_value>, G=<G_variable>)
