**Reciprocity:**
Compute the reciprocity in a directed graph.

The reciprocity of a directed graph is defined as the ratio
of the number of edges pointing in both directions to the total
number of edges in the graph.
Formally, $r = |{(u,v) \in G|(v,u) \in G}| / |{(u,v) \in G}|$.

The reciprocity of a single node u is defined similarly,
it is the ratio of the number of edges in both directions to
the total number of edges attached to node u.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --A networkx directed graph
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.reciprocity.reciprocity(G=<G_variable>)
