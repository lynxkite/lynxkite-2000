**Find cliques recursive:**
Returns all maximal cliques in a graph.

For each node *v*, a *maximal clique for v* is a largest complete
subgraph containing *v*. The largest maximal clique is sometimes
called the *maximum clique*.

This function returns an iterator over cliques, each of which is a
list of nodes. It is a recursive implementation, so may suffer from
recursion depth issues, but is included for pedagogical reasons.
For a non-recursive implementation, see :func:`find_cliques`.

This function accepts a list of `nodes` and only the maximal cliques
containing all of these `nodes` are returned. It can considerably speed up
the running time if some specific cliques are desired.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --An undirected graph.
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.clique.find_cliques_recursive(G=<G_variable>)
