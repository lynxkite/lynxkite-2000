**Transitive reduction:**
Returns transitive reduction of a directed graph

The transitive reduction of G = (V,E) is a graph G- = (V,E-) such that
for all v,w in V there is an edge (v,w) in E- if and only if (v,w) is
in E and there is no path from v to w in G with length greater than 1.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --A directed acyclic graph (DAG)

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.dag.transitive_reduction(G=<G_variable>)
