---
name: networkx-algorithms-components-weakly-connected
description: Collection of operations - Number weakly connected components, Weakly connected components, Is weakly connected
---

**Number weakly connected components:**
Returns the number of weakly connected components in G.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --A directed graph.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.components.weakly_connected.number_weakly_connected_components(G=<G_variable>)

**Weakly connected components:**
Generate weakly connected components of G.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --A directed graph

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.components.weakly_connected.weakly_connected_components(G=<G_variable>)

**Is weakly connected:**
Test directed graph for weak connectivity.

A directed graph is weakly connected if and only if the graph
is connected when the direction of the edge between nodes is ignored.

Note that if a graph is strongly connected (i.e. the graph is connected
even when we account for directionality), it is by definition weakly
connected as well.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --A directed graph.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.components.weakly_connected.is_weakly_connected(G=<G_variable>)
