**Hoffman singleton graph:**
Returns the Hoffman-Singleton Graph.

The Hoffman–Singleton graph is a symmetrical undirected graph
with 50 nodes and 175 edges.
All indices lie in ``Z % 5``: that is, the integers mod 5 [1]_.
It is the only regular graph of vertex degree 7, diameter 2, and girth 5.
It is the unique (7,5)-cage graph and Moore graph, and contains many
copies of the Petersen Graph [2]_.
parameters:


returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.small.hoffman_singleton_graph()
