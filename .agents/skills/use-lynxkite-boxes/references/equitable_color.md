**Equitable color:**
Provides an equitable coloring for nodes of `G`.

Attempts to color a graph using `num_colors` colors, where no neighbors of
a node can have same color as the node itself and the number of nodes with
each color differ by at most 1. `num_colors` must be greater than the
maximum degree of `G`. The algorithm is described in [1]_ and has
complexity O(num_colors * n**2).
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --The nodes of this graph will be colored.
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.coloring.equitable_coloring.equitable_color(G=<G_variable>)
