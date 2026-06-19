**Articulation points:**
Yield the articulation points, or cut vertices, of a graph.

An articulation point or cut vertex is any node whose removal (along with
all its incident edges) increases the number of connected components of
a graph.  An undirected connected graph without articulation points is
biconnected. Articulation points belong to more than one biconnected
component of a graph.

Notice that by convention a dyad is considered a biconnected component.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --An undirected graph.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.components.biconnected.articulation_points(G=<G_variable>)
