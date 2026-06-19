**Chordal graph cliques:**
Returns all maximal cliques of a chordal graph.

The algorithm breaks the graph in connected components and performs a
maximum cardinality search in each component to get the cliques.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --A NetworkX graph

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.chordal.chordal_graph_cliques(G=<G_variable>)
