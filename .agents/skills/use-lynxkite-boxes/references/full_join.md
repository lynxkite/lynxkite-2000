**Full join:**
Returns the full join of graphs G and H.

Full join is the union of G and H in which all edges between
G and H are added.
The node sets of G and H must be disjoint,
otherwise an exception is raised.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --A NetworkX graph
  - H: <class 'networkx.classes.graph.Graph'> = ? --A NetworkX graph

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.operators.binary.full_join(G=<G_variable>, H=<H_variable>)
