---
name: is-perfect-graph
description: Return True if G is a perfect graph, else False.
---

**Is perfect graph:**
Return True if G is a perfect graph, else False.

A graph G is perfect if, for every induced subgraph H of G, the chromatic
number of H equals the size of the largest clique in H.

According to the **Strong Perfect Graph Theorem (SPGT)**:
A graph is perfect if and only if neither the graph G nor its complement
:math:`\overline{G}` contains an **induced odd hole** — an induced cycle of
odd length at least five without chords.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.perfect_graph.is_perfect_graph(G=<G_variable>)
