---
name: percolation-centrality
description: Compute the percolation centrality for nodes.
---

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
  - attribute: str | None = percolation -
  - weight: str | None = None -
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.centrality.percolation.percolation_centrality(attribute=<attribute_value>, weight=<weight_value>, G=<G_variable>)
