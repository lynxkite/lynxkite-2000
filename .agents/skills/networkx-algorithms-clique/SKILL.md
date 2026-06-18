---
name: networkx-algorithms-clique
description: Collection of operations - Find cliques, Find cliques recursive, Make max clique graph, Enumerate all cliques, Max weight clique
---

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
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.clique.find_cliques(G=<G_variable>)

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
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.clique.find_cliques_recursive(G=<G_variable>)

**Make max clique graph:**
Returns the maximal clique graph of the given graph.

The nodes of the maximal clique graph of `G` are the cliques of
`G` and an edge joins two cliques if the cliques are not disjoint.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.clique.make_max_clique_graph(G=<G_variable>)

**Enumerate all cliques:**
Returns all cliques in an undirected graph.

This function returns an iterator over cliques, each of which is a
list of nodes. The iteration is ordered by cardinality of the
cliques: first all cliques of size one, then all cliques of size
two, etc.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.clique.enumerate_all_cliques(G=<G_variable>)

**Max weight clique:**
Find a maximum weight clique in G.

A *clique* in a graph is a set of nodes such that every two distinct nodes
are adjacent.  The *weight* of a clique is the sum of the weights of its
nodes.  A *maximum weight clique* of graph G is a clique C in G such that
no clique in G has weight greater than the weight of C.
parameters:
  - weight: <class 'int'> = weight -
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.clique.max_weight_clique(weight=<weight_value>, G=<G_variable>)
