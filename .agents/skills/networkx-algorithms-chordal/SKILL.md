---
name: networkx-algorithms-chordal
description: Collection of operations - Is chordal, Chordal graph cliques, Chordal graph treewidth, Complete to chordal graph
---

**Is chordal:**
Checks whether G is a chordal graph.

A graph is chordal if every cycle of length at least 4 has a chord
(an edge joining two nodes not adjacent in the cycle).
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.chordal.is_chordal(G=<G_variable>)

**Chordal graph cliques:**
Returns all maximal cliques of a chordal graph.

The algorithm breaks the graph in connected components and performs a
maximum cardinality search in each component to get the cliques.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.chordal.chordal_graph_cliques(G=<G_variable>)

**Chordal graph treewidth:**
Returns the treewidth of the chordal graph G.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.chordal.chordal_graph_treewidth(G=<G_variable>)

**Complete to chordal graph:**
Return a copy of G completed to a chordal graph

Adds edges to a copy of G to create a chordal graph. A graph G=(V,E) is
called chordal if for each cycle with length bigger than 3, there exist
two non-adjacent nodes connected by an edge (called a chord).
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.chordal.complete_to_chordal_graph(G=<G_variable>)
