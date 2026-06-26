**Transitive closure:**
Returns transitive closure of a graph

The transitive closure of G = (V,E) is a graph G+ = (V,E+) such that
for all v, w in V there is an edge (v, w) in E+ if and only if there
is a path from v to w in G.

Handling of paths from v to v has some flexibility within this definition.
A reflexive transitive closure creates a self-loop for the path
from v to v of length 0. The usual transitive closure creates a
self-loop only if a cycle exists (a path from v to v with length > 0).
We also allow an option for no self-loops.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --A directed/undirected graph/multigraph.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.dag.transitive_closure(G=<G_variable>)
