---
name: greedy-color
description: Color a graph using various strategies of greedy graph coloring.
---

**Greedy color:**
Color a graph using various strategies of greedy graph coloring.

Attempts to color a graph using as few colors as possible, where no
neighbors of a node can have same color as the node itself. The
given strategy determines the order in which nodes are colored.

The strategies are described in [1]_, and smallest-last is based on
[2]_.
parameters:
  - interchange: <class 'bool'> = None -
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.coloring.greedy_coloring.greedy_color(interchange=<interchange_value>, G=<G_variable>)
