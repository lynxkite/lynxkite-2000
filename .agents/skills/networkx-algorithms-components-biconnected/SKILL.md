---
name: networkx-algorithms-components-biconnected
description: Collection of operations - Biconnected components, Biconnected component edges, Is biconnected, Articulation points
---

**Biconnected components:**
Returns a generator of sets of nodes, one set for each biconnected
component of the graph

Biconnected components are maximal subgraphs such that the removal of a
node (and all edges incident on that node) will not disconnect the
subgraph. Note that nodes may be part of more than one biconnected
component.  Those nodes are articulation points, or cut vertices.  The
removal of articulation points will increase the number of connected
components of the graph.

Notice that by convention a dyad is considered a biconnected component.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.components.biconnected.biconnected_components(G=<G_variable>)

**Biconnected component edges:**
Returns a generator of lists of edges, one list for each biconnected
component of the input graph.

Biconnected components are maximal subgraphs such that the removal of a
node (and all edges incident on that node) will not disconnect the
subgraph.  Note that nodes may be part of more than one biconnected
component.  Those nodes are articulation points, or cut vertices.
However, each edge belongs to one, and only one, biconnected component.

Notice that by convention a dyad is considered a biconnected component.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.components.biconnected.biconnected_component_edges(G=<G_variable>)

**Is biconnected:**
Returns True if the graph is biconnected, False otherwise.

A graph is biconnected if, and only if, it cannot be disconnected by
removing only one node (and all edges incident on that node).  If
removing a node increases the number of disconnected components
in the graph, that node is called an articulation point, or cut
vertex.  A biconnected graph has no articulation points.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.components.biconnected.is_biconnected(G=<G_variable>)

**Articulation points:**
Yield the articulation points, or cut vertices, of a graph.

An articulation point or cut vertex is any node whose removal (along with
all its incident edges) increases the number of connected components of
a graph.  An undirected connected graph without articulation points is
biconnected. Articulation points belong to more than one biconnected
component of a graph.

Notice that by convention a dyad is considered a biconnected component.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.components.biconnected.articulation_points(G=<G_variable>)
