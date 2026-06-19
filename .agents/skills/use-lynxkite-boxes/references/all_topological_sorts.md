**All topological sorts:**
Returns a generator of _all_ topological sorts of the directed graph G.

A topological sort is a nonunique permutation of the nodes such that an
edge from u to v implies that u appears before v in the topological sort
order.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --A directed graph

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.dag.all_topological_sorts(G=<G_variable>)
