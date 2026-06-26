**Power:**
Returns the specified power of a graph.

The $k$th power of a simple graph $G$, denoted $G^k$, is a
graph on the same set of nodes in which two distinct nodes $u$ and
$v$ are adjacent in $G^k$ if and only if the shortest path
distance between $u$ and $v$ in $G$ is at most $k$.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --A NetworkX simple graph object.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.operators.product.power(G=<G_variable>)
