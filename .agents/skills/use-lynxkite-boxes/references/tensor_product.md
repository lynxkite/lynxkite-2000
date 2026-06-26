**Tensor product:**
Returns the tensor product of G and H.

The tensor product $P$ of the graphs $G$ and $H$ has a node set that
is the Cartesian product of the node sets, $V(P)=V(G) \times V(H)$.
$P$ has an edge $((u,v), (x,y))$ if and only if $(u,x)$ is an edge in $G$
and $(v,y)$ is an edge in $H$.

Tensor product is sometimes also referred to as the categorical product,
direct product, cardinal product or conjunction.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --Networkx graphs.
  - H: <class 'networkx.classes.graph.Graph'> = ? --Networkx graphs.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.operators.product.tensor_product(G=<G_variable>, H=<H_variable>)
