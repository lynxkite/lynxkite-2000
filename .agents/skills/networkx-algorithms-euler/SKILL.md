---
name: networkx-algorithms-euler
description: Collection of operations - Is Eulerian, Eulerian circuit, Eulerize, Has Eulerian path, Eulerian path
---

**Is Eulerian:**
Returns True if and only if `G` is Eulerian.

A graph is *Eulerian* if it has an Eulerian circuit. An *Eulerian
circuit* is a closed walk that includes each edge of a graph exactly
once.

Graphs with isolated vertices (i.e. vertices with zero degree) are not
considered to have Eulerian circuits. Therefore, if the graph is not
connected (or not strongly connected, for directed graphs), this function
returns False.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.euler.is_eulerian(G=<G_variable>)

**Eulerian circuit:**
Returns an iterator over the edges of an Eulerian circuit in `G`.

An *Eulerian circuit* is a closed walk that includes each edge of a
graph exactly once.
parameters:
  - keys: <class 'bool'> = None -
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.euler.eulerian_circuit(keys=<keys_value>, G=<G_variable>)

**Eulerize:**
Transforms a graph into an Eulerian graph.

If `G` is Eulerian the result is `G` as a MultiGraph, otherwise the result is a smallest
(in terms of the number of edges) multigraph whose underlying simple graph is `G`.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.euler.eulerize(G=<G_variable>)

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
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.euler.has_eulerian_path(G=<G_variable>)

**Eulerian path:**
Return an iterator over the edges of an Eulerian path in `G`.
parameters:
  - keys: <class 'bool'> = None -
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.euler.eulerian_path(keys=<keys_value>, G=<G_variable>)
