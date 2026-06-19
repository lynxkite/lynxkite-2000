**Lexicographic product:**
Returns the lexicographic product of G and H.

The lexicographical product $P$ of the graphs $G$ and $H$ has a node set
that is the Cartesian product of the node sets, $V(P)=V(G) \times V(H)$.
$P$ has an edge $((u,v), (x,y))$ if and only if $(u,v)$ is an edge in $G$
or $u==v$ and $(x,y)$ is an edge in $H$.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --Networkx graphs.
  - H: <class 'networkx.classes.graph.Graph'> = ? --Networkx graphs.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.operators.product.lexicographic_product(G=<G_variable>, H=<H_variable>)
