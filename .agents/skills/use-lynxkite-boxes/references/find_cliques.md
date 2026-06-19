**Find cliques:**
Returns all maximal cliques in an undirected graph.

For each node *n*, a *maximal clique for n* is a largest complete
subgraph containing *n*. The largest maximal clique is sometimes
called the *maximum clique*.

This function returns an iterator over cliques, each of which is a
list of nodes. It is an iterative implementation, so should not
suffer from recursion depth issues.

This function accepts a list of `nodes` and only the maximal cliques
containing all of these `nodes` are returned. It can considerably speed up
the running time if some specific cliques are desired.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --An undirected graph.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.clique.find_cliques(G=<G_variable>)
