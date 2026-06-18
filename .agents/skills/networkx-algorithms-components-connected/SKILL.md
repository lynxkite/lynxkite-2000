---
name: networkx-algorithms-components-connected
description: Collection of operations - Number connected components, Connected components, Is connected, Node connected component
---

**Number connected components:**
Returns the number of connected components.

The connected components of an undirected graph partition the graph into
disjoint sets of nodes. Each of these sets induces a subgraph of graph
`G` that is connected and not part of any larger connected subgraph.

A graph is connected (:func:`is_connected`) if, for every pair of distinct
nodes, there is a path between them. If there is a pair of nodes for
which such path does not exist, the graph is not connected (also referred
to as "disconnected").

A graph consisting of a single node and no edges is connected.
Connectivity is undefined for the null graph (graph with no nodes).
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.components.connected.number_connected_components(G=<G_variable>)

**Connected components:**
Generate connected components.

The connected components of an undirected graph partition the graph into
disjoint sets of nodes. Each of these sets induces a subgraph of graph
`G` that is connected and not part of any larger connected subgraph.

A graph is connected (:func:`is_connected`) if, for every pair of distinct
nodes, there is a path between them. If there is a pair of nodes for
which such path does not exist, the graph is not connected (also referred
to as "disconnected").

A graph consisting of a single node and no edges is connected.
Connectivity is undefined for the null graph (graph with no nodes).
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.components.connected.connected_components(G=<G_variable>)

**Is connected:**
Returns True if the graph is connected, False otherwise.

A graph is connected if, for every pair of distinct nodes, there is a
path between them. If there is a pair of nodes for which such path does
not exist, the graph is not connected (also referred to as "disconnected").

A graph consisting of a single node and no edges is connected.
Connectivity is undefined for the null graph (graph with no nodes).
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.components.connected.is_connected(G=<G_variable>)

**Node connected component:**
Returns the set of nodes in the component of graph containing node n.

A connected component is a set of nodes that induces a subgraph of graph
`G` that is connected and not part of any larger connected subgraph.

A graph is connected (:func:`is_connected`) if, for every pair of distinct
nodes, there is a path between them. If there is a pair of nodes for
which such path does not exist, the graph is not connected (also referred
to as "disconnected").

A graph consisting of a single node and no edges is connected.
Connectivity is undefined for the null graph (graph with no nodes).
parameters:
  - n: <class 'str'> = None -
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.components.connected.node_connected_component(n=<n_value>, G=<G_variable>)
