---
name: networkx-algorithms-tree-recognition
description: Collection of operations - Is arborescence, Is branching, Is forest, Is tree
---

**Is arborescence:**
Returns True if `G` is an arborescence.

An arborescence is a directed tree with maximum in-degree equal to 1.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.algorithms.tree.recognition.is_arborescence(G=<G_variable>)

**Is branching:**
Returns True if `G` is a branching.

A branching is a directed forest with maximum in-degree equal to 1.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.algorithms.tree.recognition.is_branching(G=<G_variable>)

**Is forest:**
Returns True if `G` is a forest.

A forest is a graph with no undirected cycles.

For directed graphs, `G` is a forest if the underlying graph is a forest.
The underlying graph is obtained by treating each directed edge as a single
undirected edge in a multigraph.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.algorithms.tree.recognition.is_forest(G=<G_variable>)

**Is tree:**
Returns True if `G` is a tree.

A tree is a connected graph with no undirected cycles.

For directed graphs, `G` is a tree if the underlying graph is a tree. The
underlying graph is obtained by treating each directed edge as a single
undirected edge in a multigraph.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.algorithms.tree.recognition.is_tree(G=<G_variable>)
