**Has Eulerian path:**
Return True iff `G` has an Eulerian path.

An Eulerian path is a path in a graph which uses each edge of a graph
exactly once. If `source` is specified, then this function checks
whether an Eulerian path that starts at node `source` exists.

A directed graph has an Eulerian path iff:
    - at most one vertex has out_degree - in_degree = 1,
    - at most one vertex has in_degree - out_degree = 1,
    - every other vertex has equal in_degree and out_degree,
    - and all of its vertices belong to a single connected
      component of the underlying undirected graph.

If `source` is not None, an Eulerian path starting at `source` exists if no
other node has out_degree - in_degree = 1. This is equivalent to either
there exists an Eulerian circuit or `source` has out_degree - in_degree = 1
and the conditions above hold.

An undirected graph has an Eulerian path iff:
    - exactly zero or two vertices have odd degree,
    - and all of its vertices belong to a single connected component.

If `source` is not None, an Eulerian path starting at `source` exists if
either there exists an Eulerian circuit or `source` has an odd degree and the
conditions above hold.

Graphs with isolated vertices (i.e. vertices with zero degree) are not considered
to have an Eulerian path. Therefore, if the graph is not connected (or not strongly
connected, for directed graphs), this function returns False.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --The graph to find an euler path in.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.euler.has_eulerian_path(G=<G_variable>)
