**Max weight clique:**
Find a maximum weight clique in G.

A *clique* in a graph is a set of nodes such that every two distinct nodes
are adjacent.  The *weight* of a clique is the sum of the weights of its
nodes.  A *maximum weight clique* of graph G is a clique C in G such that
no clique in G has weight greater than the weight of C.
parameters:
  - weight: <class 'int'> = weight --The node attribute that holds the integer value used as a weight.
If None, then each node has weight 1.
  - G: <class 'networkx.classes.graph.Graph'> = ? --Undirected graph
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.clique.max_weight_clique(weight=<weight_value>, G=<G_variable>)
