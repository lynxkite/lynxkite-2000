**Windmill graph:**
Generate a windmill graph.
A windmill graph is a graph of `n` cliques each of size `k` that are all
joined at one node.
It can be thought of as taking a disjoint union of `n` cliques of size `k`,
selecting one point from each, and contracting all of the selected points.
Alternatively, one could generate `n` cliques of size `k-1` and one node
that is connected to all other nodes in the graph.
parameters:
  - n: <class 'int'> = ? --Number of cliques
  - k: <class 'int'> = ? --Size of cliques
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.generators.community.windmill_graph(n=<n_value>, k=<k_value>)
