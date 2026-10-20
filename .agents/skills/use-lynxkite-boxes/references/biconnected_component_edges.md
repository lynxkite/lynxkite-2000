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
  - G: <class 'networkx.classes.graph.Graph'> = ? --An undirected graph.
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.components.biconnected.biconnected_component_edges(G=<G_variable>)
