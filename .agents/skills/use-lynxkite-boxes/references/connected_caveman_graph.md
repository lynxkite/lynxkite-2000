**Connected caveman graph:**
Returns a connected caveman graph of `l` cliques of size `k`.

The connected caveman graph is formed by creating `n` cliques of size
`k`, then a single edge in each clique is rewired to a node in an
adjacent clique.
parameters:
  - l: <class 'int'> = ? --number of cliques
  - k: <class 'int'> = ? --size of cliques (k at least 2 or NetworkXError is raised)

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.community.connected_caveman_graph(l=<l_value>, k=<k_value>)
