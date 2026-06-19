---
name: networkx-algorithms-asteroidal
description: Collection of operations - Is AT-free, Find asteroidal triple
---

**Is AT-free:**
Check if a graph is AT-free.

The method uses the `find_asteroidal_triple` method to recognize
an AT-free graph. If no asteroidal triple is found the graph is
AT-free and True is returned. If at least one asteroidal triple is
found the graph is not AT-free and False is returned.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --The graph to check whether is AT-free or not.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.asteroidal.is_at_free(G=<G_variable>)

**Find asteroidal triple:**
Find an asteroidal triple in the given graph.

An asteroidal triple is a triple of non-adjacent vertices such that
there exists a path between any two of them which avoids the closed
neighborhood of the third. It checks all independent triples of vertices
and whether they are an asteroidal triple or not. This is done with the
help of a data structure called a component structure.
A component structure encodes information about which vertices belongs to
the same connected component when the closed neighborhood of a given vertex
is removed from the graph. The algorithm used to check is the trivial
one, outlined in [1]_, which has a runtime of
:math:`O(|V||\overline{E} + |V||E|)`, where the second term is the
creation of the component structure.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --The graph to check whether is AT-free or not

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.asteroidal.find_asteroidal_triple(G=<G_variable>)
