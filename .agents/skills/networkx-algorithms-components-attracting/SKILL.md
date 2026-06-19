---
name: networkx-algorithms-components-attracting
description: Collection of operations - Number attracting components, Attracting components, Is attracting component
---

**Number attracting components:**
Returns the number of attracting components in `G`.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --The graph to be analyzed.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.components.attracting.number_attracting_components(G=<G_variable>)

**Attracting components:**
Generates the attracting components in `G`.

An attracting component in a directed graph `G` is a strongly connected
component with the property that a random walker on the graph will never
leave the component, once it enters the component.

The nodes in attracting components can also be thought of as recurrent
nodes.  If a random walker enters the attractor containing the node, then
the node will be visited infinitely often.

To obtain induced subgraphs on each component use:
``(G.subgraph(c).copy() for c in attracting_components(G))``
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --The graph to be analyzed.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.components.attracting.attracting_components(G=<G_variable>)

**Is attracting component:**
Returns True if `G` consists of a single attracting component.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --The graph to be analyzed.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.components.attracting.is_attracting_component(G=<G_variable>)
