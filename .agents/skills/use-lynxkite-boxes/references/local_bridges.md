**Local bridges:**
Iterate over local bridges of `G` optionally computing the span

A *local bridge* is an edge whose endpoints have no common neighbors.
That is, the edge is not part of a triangle in the graph.

The *span* of a *local bridge* is the shortest path length between
the endpoints if the local bridge is removed.
parameters:
  - with_span: <class 'bool'> = ? --If True, yield a 3-tuple `(u, v, span)`
  - weight: <class 'str'> = ? --If function, used to compute edge weights for the span.
If string, the edge data attribute used in calculating span.
If None, all edges have weight 1.
  - G: <class 'networkx.classes.graph.Graph'> = ? --?
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.bridges.local_bridges(with_span=<with_span_value>, weight=<weight_value>, G=<G_variable>)
