**Percolation centrality:**
Compute the percolation centrality for nodes.

Percolation centrality of a node $v$, at a given time, is defined
as the proportion of ‘percolated paths’ that go through that node.

This measure quantifies relative impact of nodes based on their
topological connectivity, as well as their percolation states.

Percolation states of nodes are used to depict network percolation
scenarios (such as during infection transmission in a social network
of individuals, spreading of computer viruses on computer networks, or
transmission of disease over a network of towns) over time. In this
measure usually the percolation state is expressed as a decimal
between 0.0 and 1.0.

When all nodes are in the same percolated state this measure is
equivalent to betweenness centrality.
parameters:
  - attribute: str | None = percolation --Name of the node attribute to use for percolation state, used
if `states` is None. If a node does not set the attribute the
state of that node will be set to the default value of 1.
If all nodes do not have the attribute all nodes will be set to
1 and the centrality measure will be equivalent to betweenness centrality.
  - weight: str | None = ? --If None, all edge weights are considered equal.
Otherwise holds the name of the edge attribute used as weight.
The weight of an edge is treated as the length or distance between the two sides.
  - G: <class 'networkx.classes.graph.Graph'> = ? --A NetworkX graph.
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.centrality.percolation.percolation_centrality(attribute=<attribute_value>, weight=<weight_value>, G=<G_variable>)
