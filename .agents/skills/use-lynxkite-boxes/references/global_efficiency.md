**Global efficiency:**
Returns the average global efficiency of the graph.

The *efficiency* of a pair of nodes in a graph is the multiplicative
inverse of the shortest path distance between the nodes. The *average
global efficiency* of a graph is the average efficiency of all pairs of
nodes [1]_.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --An undirected graph for which to compute the average global efficiency.
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.efficiency_measures.global_efficiency(G=<G_variable>)
