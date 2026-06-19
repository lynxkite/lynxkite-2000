---
name: networkx-algorithms-dag
description: Collection of operations - Descendants, Ancestors, Topological sort, Lexicographical topological sort, All topological sorts, Topological generations, Is directed acyclic graph, Is aperiodic, Transitive closure, Transitive closure DAG, Transitive reduction, Antichains, DAG longest path, DAG longest path length, DAG to branching
---

**Descendants:**
Returns all nodes reachable from `source` in `G`.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --?

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.dag.descendants(G=<G_variable>)

**Ancestors:**
Returns all nodes having a path to `source` in `G`.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --?

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.dag.ancestors(G=<G_variable>)

**Topological sort:**
Returns a generator of nodes in topologically sorted order.

A topological sort is a nonunique permutation of the nodes of a
directed graph such that an edge from u to v implies that u
appears before v in the topological sort order. This ordering is
valid only if the graph has no directed cycles.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --A directed acyclic graph (DAG)

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.dag.topological_sort(G=<G_variable>)

**Lexicographical topological sort:**
Generate the nodes in the unique lexicographical topological sort order.

Generates a unique ordering of nodes by first sorting topologically (for which there are often
multiple valid orderings) and then additionally by sorting lexicographically.

A topological sort arranges the nodes of a directed graph so that the
upstream node of each directed edge precedes the downstream node.
It is always possible to find a solution for directed graphs that have no cycles.
There may be more than one valid solution.

Lexicographical sorting is just sorting alphabetically. It is used here to break ties in the
topological sort and to determine a single, unique ordering.  This can be useful in comparing
sort results.

The lexicographical order can be customized by providing a function to the `key=` parameter.
The definition of the key function is the same as used in python's built-in `sort()`.
The function takes a single argument and returns a key to use for sorting purposes.

Lexicographical sorting can fail if the node names are un-sortable. See the example below.
The solution is to provide a function to the `key=` argument that returns sortable keys.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --A directed acyclic graph (DAG)

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.dag.lexicographical_topological_sort(G=<G_variable>)

**All topological sorts:**
Returns a generator of _all_ topological sorts of the directed graph G.

A topological sort is a nonunique permutation of the nodes such that an
edge from u to v implies that u appears before v in the topological sort
order.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --A directed graph

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.dag.all_topological_sorts(G=<G_variable>)

**Topological generations:**
Stratifies a DAG into generations.

A topological generation is node collection in which ancestors of a node in each
generation are guaranteed to be in a previous generation, and any descendants of
a node are guaranteed to be in a following generation. Nodes are guaranteed to
be in the earliest possible generation that they can belong to.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --A directed acyclic graph (DAG)

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.dag.topological_generations(G=<G_variable>)

**Is directed acyclic graph:**
Returns True if the graph `G` is a directed acyclic graph (DAG) or
False if not.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --?

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.dag.is_directed_acyclic_graph(G=<G_variable>)

**Is aperiodic:**
Returns True if `G` is aperiodic.

A strongly connected directed graph is aperiodic if there is no integer ``k > 1``
that divides the length of every cycle in the graph.

This function requires the graph `G` to be strongly connected and will raise
an error if it's not. For graphs that are not strongly connected, you should
first identify their strongly connected components
(using :func:`~networkx.algorithms.components.strongly_connected_components`)
or attracting components
(using :func:`~networkx.algorithms.components.attracting_components`),
and then apply this function to those individual components.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --A directed graph

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.dag.is_aperiodic(G=<G_variable>)

**Transitive closure:**
Returns transitive closure of a graph

The transitive closure of G = (V,E) is a graph G+ = (V,E+) such that
for all v, w in V there is an edge (v, w) in E+ if and only if there
is a path from v to w in G.

Handling of paths from v to v has some flexibility within this definition.
A reflexive transitive closure creates a self-loop for the path
from v to v of length 0. The usual transitive closure creates a
self-loop only if a cycle exists (a path from v to v with length > 0).
We also allow an option for no self-loops.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --A directed/undirected graph/multigraph.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.dag.transitive_closure(G=<G_variable>)

**Transitive closure DAG:**
Returns the transitive closure of a directed acyclic graph.

This function is faster than the function `transitive_closure`, but fails
if the graph has a cycle.

The transitive closure of G = (V,E) is a graph G+ = (V,E+) such that
for all v, w in V there is an edge (v, w) in E+ if and only if there
is a non-null path from v to w in G.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --A directed acyclic graph (DAG)

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.dag.transitive_closure_dag(G=<G_variable>)

**Transitive reduction:**
Returns transitive reduction of a directed graph

The transitive reduction of G = (V,E) is a graph G- = (V,E-) such that
for all v,w in V there is an edge (v,w) in E- if and only if (v,w) is
in E and there is no path from v to w in G with length greater than 1.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --A directed acyclic graph (DAG)

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.dag.transitive_reduction(G=<G_variable>)

**Antichains:**
Generates antichains from a directed acyclic graph (DAG).

An antichain is a subset of a partially ordered set such that any
two elements in the subset are incomparable.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --A directed acyclic graph (DAG)

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.dag.antichains(G=<G_variable>)

**DAG longest path:**
Returns the longest path in a directed acyclic graph (DAG).

If `G` has edges with `weight` attribute the edge data are used as
weight values.
parameters:
  - weight: str | None = weight --Edge data key to use for weight
  - default_weight: int | None = 1 --The weight of edges that do not have a weight attribute
  - G: <class 'networkx.classes.graph.Graph'> = ? --A directed acyclic graph (DAG)

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.dag.dag_longest_path(weight=<weight_value>, default_weight=<default_weight_value>, G=<G_variable>)

**DAG longest path length:**
Returns the longest path length in a DAG
parameters:
  - weight: str | None = weight --Edge data key to use for weight
  - default_weight: int | None = 1 --The weight of edges that do not have a weight attribute
  - G: <class 'networkx.classes.graph.Graph'> = ? --A directed acyclic graph (DAG)

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.dag.dag_longest_path_length(weight=<weight_value>, default_weight=<default_weight_value>, G=<G_variable>)

**DAG to branching:**
Returns a branching representing all (overlapping) paths from
root nodes to leaf nodes in the given directed acyclic graph.

As described in :mod:`networkx.algorithms.tree.recognition`, a
*branching* is a directed forest in which each node has at most one
parent. In other words, a branching is a disjoint union of
*arborescences*. For this function, each node of in-degree zero in
`G` becomes a root of one of the arborescences, and there will be
one leaf node for each distinct path from that root to a leaf node
in `G`.

Each node `v` in `G` with *k* parents becomes *k* distinct nodes in
the returned branching, one for each parent, and the sub-DAG rooted
at `v` is duplicated for each copy. The algorithm then recurses on
the children of each copy of `v`.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --A directed acyclic graph.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.dag.dag_to_branching(G=<G_variable>)
