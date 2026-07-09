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
  - G: <class 'networkx.classes.graph.Graph'> = ? --A NetworkX Graph, DiGraph, MultiGraph, or MultiDigraph.
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.generators.line.line_graph(G=<G_variable>)
