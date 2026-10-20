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
  - G: <class 'networkx.classes.graph.Graph'> = ? --?
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.polynomials.chromatic_polynomial(G=<G_variable>)
