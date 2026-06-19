---
name: networkx-algorithms-operators-product
description: Collection of operations - Tensor product, Cartesian product, Lexicographic product, Strong product, Power, Corona product, Modular product
---

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
