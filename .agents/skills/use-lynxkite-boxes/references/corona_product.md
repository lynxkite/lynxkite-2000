**Corona product:**
Returns the Corona product of G and H.

The corona product of $G$ and $H$ is the graph $C = G \circ H$ obtained by
taking one copy of $G$, called the center graph, $|V(G)|$ copies of $H$,
called the outer graph, and making the $i$-th vertex of $G$ adjacent to
every vertex of the $i$-th copy of $H$, where $1 ≤ i ≤ |V(G)|$.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --The graphs to take the carona product of.
`G` is the center graph and `H` is the outer graph
  - H: <class 'networkx.classes.graph.Graph'> = ? --The graphs to take the carona product of.
`G` is the center graph and `H` is the outer graph
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.operators.product.corona_product(G=<G_variable>, H=<H_variable>)
