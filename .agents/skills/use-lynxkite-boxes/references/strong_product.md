**Strong product:**
Returns the strong product of G and H.

The strong product $P$ of the graphs $G$ and $H$ has a node set that
is the Cartesian product of the node sets, $V(P)=V(G) \times V(H)$.
$P$ has an edge $((u,x), (v,y))$ if any of the following conditions
are met:

- $u=v$ and $(x,y)$ is an edge in $H$
- $x=y$ and $(u,v)$ is an edge in $G$
- $(u,v)$ is an edge in $G$ and $(x,y)$ is an edge in $H$
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --Networkx graphs.
  - H: <class 'networkx.classes.graph.Graph'> = ? --Networkx graphs.
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.operators.product.strong_product(G=<G_variable>, H=<H_variable>)
