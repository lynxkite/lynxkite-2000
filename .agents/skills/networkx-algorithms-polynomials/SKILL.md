---
name: networkx-algorithms-polynomials
description: Collection of operations - Tutte polynomial, Chromatic polynomial
---

**Tutte polynomial:**
Returns the Tutte polynomial of `G`

This function computes the Tutte polynomial via an iterative version of
the deletion-contraction algorithm.

The Tutte polynomial `T_G(x, y)` is a fundamental graph polynomial invariant in
two variables. It encodes a wide array of information related to the
edge-connectivity of a graph; "Many problems about graphs can be reduced to
problems of finding and evaluating the Tutte polynomial at certain values" [1]_.
In fact, every deletion-contraction-expressible feature of a graph is a
specialization of the Tutte polynomial [2]_ (see Notes for examples).

There are several equivalent definitions; here are three:

Def 1 (rank-nullity expansion): For `G` an undirected graph, `n(G)` the
number of vertices of `G`, `E` the edge set of `G`, `V` the vertex set of
`G`, and `c(A)` the number of connected components of the graph with vertex
set `V` and edge set `A` [3]_:

.. math::

    T_G(x, y) = \sum_{A \in E} (x-1)^{c(A) - c(E)} (y-1)^{c(A) + |A| - n(G)}

Def 2 (spanning tree expansion): Let `G` be an undirected graph, `T` a spanning
tree of `G`, and `E` the edge set of `G`. Let `E` have an arbitrary strict
linear order `L`. Let `B_e` be the unique minimal nonempty edge cut of
$E \setminus T \cup {e}$. An edge `e` is internally active with respect to
`T` and `L` if `e` is the least edge in `B_e` according to the linear order
`L`. The internal activity of `T` (denoted `i(T)`) is the number of edges
in $E \setminus T$ that are internally active with respect to `T` and `L`.
Let `P_e` be the unique path in $T \cup {e}$ whose source and target vertex
are the same. An edge `e` is externally active with respect to `T` and `L`
if `e` is the least edge in `P_e` according to the linear order `L`. The
external activity of `T` (denoted `e(T)`) is the number of edges in
$E \setminus T$ that are externally active with respect to `T` and `L`.
Then [4]_ [5]_:

.. math::

    T_G(x, y) = \sum_{T \text{ a spanning tree of } G} x^{i(T)} y^{e(T)}

Def 3 (deletion-contraction recurrence): For `G` an undirected graph, `G-e`
the graph obtained from `G` by deleting edge `e`, `G/e` the graph obtained
from `G` by contracting edge `e`, `k(G)` the number of cut-edges of `G`,
and `l(G)` the number of self-loops of `G`:

.. math::
    T_G(x, y) = \begin{cases}
       x^{k(G)} y^{l(G)}, & \text{if all edges are cut-edges or self-loops} \\
       T_{G-e}(x, y) + T_{G/e}(x, y), & \text{otherwise, for an arbitrary edge $e$ not a cut-edge or loop}
    \end{cases}
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.algorithms.polynomials.tutte_polynomial(G=<G_variable>)

**Chromatic polynomial:**
Returns the chromatic polynomial of `G`

This function computes the chromatic polynomial via an iterative version of
the deletion-contraction algorithm.

The chromatic polynomial `X_G(x)` is a fundamental graph polynomial
invariant in one variable. Evaluating `X_G(k)` for an natural number `k`
enumerates the proper k-colorings of `G`.

There are several equivalent definitions; here are three:

Def 1 (explicit formula):
For `G` an undirected graph, `c(G)` the number of connected components of
`G`, `E` the edge set of `G`, and `G(S)` the spanning subgraph of `G` with
edge set `S` [1]_:

.. math::

    X_G(x) = \sum_{S \subseteq E} (-1)^{|S|} x^{c(G(S))}


Def 2 (interpolating polynomial):
For `G` an undirected graph, `n(G)` the number of vertices of `G`, `k_0 = 0`,
and `k_i` the number of distinct ways to color the vertices of `G` with `i`
unique colors (for `i` a natural number at most `n(G)`), `X_G(x)` is the
unique Lagrange interpolating polynomial of degree `n(G)` through the points
`(0, k_0), (1, k_1), \dots, (n(G), k_{n(G)})` [2]_.


Def 3 (chromatic recurrence):
For `G` an undirected graph, `G-e` the graph obtained from `G` by deleting
edge `e`, `G/e` the graph obtained from `G` by contracting edge `e`, `n(G)`
the number of vertices of `G`, and `e(G)` the number of edges of `G` [3]_:

.. math::
    X_G(x) = \begin{cases}
       x^{n(G)}, & \text{if $e(G)=0$} \\
       X_{G-e}(x) - X_{G/e}(x), & \text{otherwise, for an arbitrary edge $e$}
    \end{cases}

This formulation is also known as the Fundamental Reduction Theorem [4]_.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.algorithms.polynomials.chromatic_polynomial(G=<G_variable>)
