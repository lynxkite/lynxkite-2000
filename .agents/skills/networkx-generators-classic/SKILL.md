---
name: networkx-generators-classic
description: Collection of operations - Balanced tree, Barbell graph, Binomial tree, Complete graph, Complete multipartite graph, Circular ladder graph, Circulant graph, Cycle graph, Dorogovtsev–Goltsev–Mendes graph, Empty graph, Full r-ary tree, Kneser graph, Ladder graph, Lollipop graph, Null graph, Path graph, Star graph, Tadpole graph, Trivial graph, Turan graph, Wheel graph
---

**Balanced tree:**
Returns the perfectly balanced `r`-ary tree of height `h`.

.. plot::

    >>> nx.draw(nx.balanced_tree(2, 3))
parameters:
  - r: <class 'int'> = None -
  - h: <class 'int'> = None -

usage:
output_variable = networkx.generators.classic.balanced_tree(r=<r_value>, h=<h_value>)

**Barbell graph:**
Returns the Barbell Graph: two complete graphs connected by a path.

.. plot::

    >>> nx.draw(nx.barbell_graph(4, 2))
parameters:
  - m1: <class 'int'> = None -
  - m2: <class 'int'> = None -

usage:
output_variable = networkx.generators.classic.barbell_graph(m1=<m1_value>, m2=<m2_value>)

**Binomial tree:**
Returns the Binomial Tree of order n.

The binomial tree of order 0 consists of a single node. A binomial tree of order k
is defined recursively by linking two binomial trees of order k-1: the root of one is
the leftmost child of the root of the other.

.. plot::

    >>> nx.draw(nx.binomial_tree(3))
parameters:
  - n: <class 'int'> = None -

usage:
output_variable = networkx.generators.classic.binomial_tree(n=<n_value>)

**Complete graph:**
Return the complete graph `K_n` with n nodes.

A complete graph on `n` nodes means that all pairs
of distinct nodes have an edge connecting them.

.. plot::

    >>> nx.draw(nx.complete_graph(5))
parameters:


usage:
output_variable = networkx.generators.classic.complete_graph()

**Complete multipartite graph:**
Returns the complete multipartite graph with the specified subset sizes.

.. plot::

    >>> nx.draw(nx.complete_multipartite_graph(1, 2, 3))
parameters:


usage:
output_variable = networkx.generators.classic.complete_multipartite_graph()

**Circular ladder graph:**
Returns the circular ladder graph $CL_n$ of length n.

$CL_n$ consists of two concentric n-cycles in which
each of the n pairs of concentric nodes are joined by an edge.

Node labels are the integers 0 to n-1

.. plot::

    >>> nx.draw(nx.circular_ladder_graph(5))
parameters:
  - n: <class 'int'> = None -

usage:
output_variable = networkx.generators.classic.circular_ladder_graph(n=<n_value>)

**Circulant graph:**
Returns the circulant graph $Ci_n(x_1, x_2, ..., x_m)$ with $n$ nodes.

The circulant graph $Ci_n(x_1, ..., x_m)$ consists of $n$ nodes $0, ..., n-1$
such that node $i$ is connected to nodes $(i + x) \mod n$ and $(i - x) \mod n$
for all $x$ in $x_1, ..., x_m$. Thus $Ci_n(1)$ is a cycle graph.

.. plot::

    >>> nx.draw(nx.circulant_graph(10, [1]))
parameters:
  - n: <class 'int'> = None -

usage:
output_variable = networkx.generators.classic.circulant_graph(n=<n_value>)

**Cycle graph:**
Returns the cycle graph $C_n$ of cyclically connected nodes.

$C_n$ is a path with its two end-nodes connected.

.. plot::

    >>> nx.draw(nx.cycle_graph(5))
parameters:


usage:
output_variable = networkx.generators.classic.cycle_graph()

**Dorogovtsev–Goltsev–Mendes graph:**
Returns the hierarchically constructed Dorogovtsev--Goltsev--Mendes graph.

The Dorogovtsev--Goltsev--Mendes [1]_ procedure deterministically produces a
scale-free graph with ``3/2 * (3**(n-1) + 1)`` nodes
and ``3**n`` edges for a given `n`.

Note that `n` denotes the number of times the state transition is applied,
starting from the base graph with ``n = 0`` (no transitions), as in [2]_.
This is different from the parameter ``t = n - 1`` in [1]_.

.. plot::

    >>> nx.draw(nx.dorogovtsev_goltsev_mendes_graph(3))
parameters:
  - n: <class 'int'> = None -

usage:
output_variable = networkx.generators.classic.dorogovtsev_goltsev_mendes_graph(n=<n_value>)

**Empty graph:**
Returns the empty graph with n nodes and zero edges.

.. plot::

    >>> nx.draw(nx.empty_graph(5))
parameters:


usage:
output_variable = networkx.generators.classic.empty_graph()

**Full r-ary tree:**
Creates a full r-ary tree of `n` nodes.

Sometimes called a k-ary, n-ary, or m-ary tree.
"... all non-leaf nodes have exactly r children and all levels
are full except for some rightmost position of the bottom level
(if a leaf at the bottom level is missing, then so are all of the
leaves to its right." [1]_

.. plot::

    >>> nx.draw(nx.full_rary_tree(2, 10))
parameters:
  - r: <class 'int'> = None -
  - n: <class 'int'> = None -

usage:
output_variable = networkx.generators.classic.full_rary_tree(r=<r_value>, n=<n_value>)

**Kneser graph:**
Returns the Kneser Graph with parameters `n` and `k`.

The Kneser Graph has nodes that are k-tuples (subsets) of the integers
between 0 and ``n-1``. Nodes are adjacent if their corresponding sets are disjoint.
parameters:
  - n: <class 'int'> = None -
  - k: <class 'int'> = None -

usage:
output_variable = networkx.generators.classic.kneser_graph(n=<n_value>, k=<k_value>)

**Ladder graph:**
Returns the Ladder graph of length n.

This is two paths of n nodes, with
each pair connected by a single edge.

Node labels are the integers 0 to 2*n - 1.

.. plot::

    >>> nx.draw(nx.ladder_graph(5))
parameters:
  - n: <class 'int'> = None -

usage:
output_variable = networkx.generators.classic.ladder_graph(n=<n_value>)

**Lollipop graph:**
Returns the Lollipop Graph; ``K_m`` connected to ``P_n``.

This is the Barbell Graph without the right barbell.

.. plot::

    >>> nx.draw(nx.lollipop_graph(3, 4))
parameters:


usage:
output_variable = networkx.generators.classic.lollipop_graph()

**Null graph:**
Returns the Null graph with no nodes or edges.

See empty_graph for the use of create_using.
parameters:


usage:
output_variable = networkx.generators.classic.null_graph()

**Path graph:**
Returns the Path graph `P_n` of linearly connected nodes.

.. plot::

    >>> nx.draw(nx.path_graph(5))
parameters:


usage:
output_variable = networkx.generators.classic.path_graph()

**Star graph:**
Return a star graph.

The star graph consists of one center node connected to `n` outer nodes.

.. plot::

    >>> nx.draw(nx.star_graph(6))
parameters:


usage:
output_variable = networkx.generators.classic.star_graph()

**Tadpole graph:**
Returns the (m,n)-tadpole graph; ``C_m`` connected to ``P_n``.

This graph on m+n nodes connects a cycle of size `m` to a path of length `n`.
It looks like a tadpole. It is also called a kite graph or a dragon graph.

.. plot::

    >>> nx.draw(nx.tadpole_graph(3, 5))
parameters:


usage:
output_variable = networkx.generators.classic.tadpole_graph()

**Trivial graph:**
Return the Trivial graph with one node (with label 0) and no edges.

.. plot::

    >>> nx.draw(nx.trivial_graph(), with_labels=True)
parameters:


usage:
output_variable = networkx.generators.classic.trivial_graph()

**Turan graph:**
Return the Turan Graph

The Turan Graph is a complete multipartite graph on $n$ nodes
with $r$ disjoint subsets. That is, edges connect each node to
every node not in its subset.

Given $n$ and $r$, we create a complete multipartite graph with
$r-(n \mod r)$ partitions of size $n/r$, rounded down, and
$n \mod r$ partitions of size $n/r+1$, rounded down.

.. plot::

    >>> nx.draw(nx.turan_graph(6, 2))
parameters:
  - n: <class 'int'> = None -
  - r: <class 'int'> = None -

usage:
output_variable = networkx.generators.classic.turan_graph(n=<n_value>, r=<r_value>)

**Wheel graph:**
Return the wheel graph

The wheel graph consists of a hub node connected to a cycle of (n-1) nodes.

.. plot::

    >>> nx.draw(nx.wheel_graph(5))
parameters:


usage:
output_variable = networkx.generators.classic.wheel_graph()
