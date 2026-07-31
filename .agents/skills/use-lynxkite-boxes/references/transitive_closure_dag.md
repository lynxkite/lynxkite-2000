**Transitive closure DAG:**
Returns the transitive closure of a directed acyclic graph.

This function is faster than the function `transitive_closure`, but fails
if the graph has a cycle.

The transitive closure of G = (V,E) is a graph G+ = (V,E+) such that
for all v, w in V there is an edge (v, w) in E+ if and only if there
is a non-null path from v to w in G.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --A directed acyclic graph (DAG)
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.dag.transitive_closure_dag(G=<G_variable>)
