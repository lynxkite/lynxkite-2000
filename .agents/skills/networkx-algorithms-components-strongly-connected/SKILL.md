---
name: networkx-algorithms-components-strongly-connected
description: Collection of operations - Number strongly connected components, Strongly connected components, Is strongly connected, Kosaraju strongly connected components, Condensation
---

**Number strongly connected components:**
Returns number of strongly connected components in graph.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.algorithms.components.strongly_connected.number_strongly_connected_components(G=<G_variable>)

**Strongly connected components:**
Generate nodes in strongly connected components of graph.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.algorithms.components.strongly_connected.strongly_connected_components(G=<G_variable>)

**Is strongly connected:**
Test directed graph for strong connectivity.

A directed graph is strongly connected if and only if every vertex in
the graph is reachable from every other vertex.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.algorithms.components.strongly_connected.is_strongly_connected(G=<G_variable>)

**Kosaraju strongly connected components:**
Generate nodes in strongly connected components of graph.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.algorithms.components.strongly_connected.kosaraju_strongly_connected_components(G=<G_variable>)

**Condensation:**
Returns the condensation of G.

The condensation of G is the graph with each of the strongly connected
components contracted into a single node.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.algorithms.components.strongly_connected.condensation(G=<G_variable>)
