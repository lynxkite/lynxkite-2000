**Intersection:**
Returns a new graph that contains only the nodes and the edges that exist in
both G and H.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --A NetworkX graph. G and H can have different node sets but must be both graphs or both multigraphs.
  - H: <class 'networkx.classes.graph.Graph'> = ? --?

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.operators.binary.intersection(G=<G_variable>, H=<H_variable>)
