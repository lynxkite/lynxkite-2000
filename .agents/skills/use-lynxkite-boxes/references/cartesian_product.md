**Cartesian product:**
Returns the Cartesian product of G and H.

The Cartesian product $P$ of the graphs $G$ and $H$ has a node set that
is the Cartesian product of the node sets, $V(P)=V(G) \times V(H)$.
$P$ has an edge $((u,v),(x,y))$ if and only if either $u$ is equal to $x$
and both $v$ and $y$ are adjacent in $H$ or if $v$ is equal to $y$ and
both $u$ and $x$ are adjacent in $G$.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --Networkx graphs.
  - H: <class 'networkx.classes.graph.Graph'> = ? --Networkx graphs.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.operators.product.cartesian_product(G=<G_variable>, H=<H_variable>)
