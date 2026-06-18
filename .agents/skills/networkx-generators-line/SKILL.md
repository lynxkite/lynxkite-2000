---
name: networkx-generators-line
description: Collection of operations - Line graph, Inverse line graph
---

**Line graph:**
Returns the line graph of the graph or digraph `G`.

The line graph of a graph `G` has a node for each edge in `G` and an
edge joining those nodes if the two edges in `G` share a common node. For
directed graphs, nodes are adjacent exactly when the edges they represent
form a directed path of length two.

The nodes of the line graph are 2-tuples of nodes in the original graph (or
3-tuples for multigraphs, with the key of the edge as the third element).

For information about self-loops and more discussion, see the **Notes**
section below.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.generators.line.line_graph(G=<G_variable>)

**Inverse line graph:**
Returns the inverse line graph of graph G.

If H is a graph, and G is the line graph of H, such that G = L(H).
Then H is the inverse line graph of G.

Not all graphs are line graphs and these do not have an inverse line graph.
In these cases this function raises a NetworkXError.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.generators.line.inverse_line_graph(G=<G_variable>)
