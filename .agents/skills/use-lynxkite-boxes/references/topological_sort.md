**Topological sort:**
Returns a generator of nodes in topologically sorted order.

A topological sort is a nonunique permutation of the nodes of a
directed graph such that an edge from u to v implies that u
appears before v in the topological sort order. This ordering is
valid only if the graph has no directed cycles.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --A directed acyclic graph (DAG)

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.dag.topological_sort(G=<G_variable>)
