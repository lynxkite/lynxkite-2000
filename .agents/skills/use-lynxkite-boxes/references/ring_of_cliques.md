**Ring of cliques:**
Defines a "ring of cliques" graph.

A ring of cliques graph is consisting of cliques, connected through single
links. Each clique is a complete graph.
parameters:
  - num_cliques: <class 'int'> = ? --Number of cliques
  - clique_size: <class 'int'> = ? --Size of cliques

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.community.ring_of_cliques(num_cliques=<num_cliques_value>, clique_size=<clique_size_value>)
