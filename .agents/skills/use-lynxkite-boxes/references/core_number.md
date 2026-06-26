**Core number:**
Returns the core number for each node.

A k-core is a maximal subgraph that contains nodes of degree k or more.

The core number of a node is the largest value k of a k-core containing
that node.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --An undirected or directed graph

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.core.core_number(G=<G_variable>)
