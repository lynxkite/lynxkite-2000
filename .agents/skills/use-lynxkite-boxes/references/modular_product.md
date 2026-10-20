**Modular product:**
Returns the Modular product of G and H.

The modular product of `G` and `H` is the graph $M = G \nabla H$,
consisting of the node set $V(M) = V(G) \times V(H)$ that is the Cartesian
product of the node sets of `G` and `H`. Further, M contains an edge ((u, v), (x, y)):

- if u is adjacent to x in `G` and v is adjacent to y in `H`, or
- if u is not adjacent to x in `G` and v is not adjacent to y in `H`.

More formally::

    E(M) = {((u, v), (x, y)) | ((u, x) in E(G) and (v, y) in E(H)) or
                               ((u, x) not in E(G) and (v, y) not in E(H))}
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --The graphs to take the modular product of.
  - H: <class 'networkx.classes.graph.Graph'> = ? --The graphs to take the modular product of.
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.operators.product.modular_product(G=<G_variable>, H=<H_variable>)
