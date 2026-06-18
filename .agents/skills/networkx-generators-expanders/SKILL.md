---
name: networkx-generators-expanders
description: Collection of operations - Margulis–Gabber–Galil graph, Chordal cycle graph, Paley graph, Maybe regular expander graph, Is regular expander, Random regular expander graph
---

**Margulis–Gabber–Galil graph:**
Returns the Margulis-Gabber-Galil undirected MultiGraph on `n^2` nodes.

The undirected MultiGraph is regular with degree `8`. Nodes are integer
pairs. The second-largest eigenvalue of the adjacency matrix of the graph
is at most `5 \sqrt{2}`, regardless of `n`.
parameters:
  - n: <class 'int'> = None -

usage:
output_variable = networkx.generators.expanders.margulis_gabber_galil_graph(n=<n_value>)

**Chordal cycle graph:**
Returns the chordal cycle graph on `p` nodes.

The returned graph is a cycle graph on `p` nodes with chords joining each
vertex `x` to its inverse modulo `p`. This graph is a (mildly explicit)
3-regular expander [1]_.

`p` *must* be a prime number.
parameters:


usage:
output_variable = networkx.generators.expanders.chordal_cycle_graph()

**Paley graph:**
Returns the Paley $\frac{(p-1)}{2}$ -regular graph on $p$ nodes.

The returned graph is a graph on $\mathbb{Z}/p\mathbb{Z}$ with edges between $x$ and $y$
if and only if $x-y$ is a nonzero square in $\mathbb{Z}/p\mathbb{Z}$.

If $p \equiv 1  \pmod 4$, $-1$ is a square in
$\mathbb{Z}/p\mathbb{Z}$ and therefore $x-y$ is a square if and
only if $y-x$ is also a square, i.e the edges in the Paley graph are symmetric.

If $p \equiv 3 \pmod 4$, $-1$ is not a square in $\mathbb{Z}/p\mathbb{Z}$
and therefore either $x-y$ or $y-x$ is a square in $\mathbb{Z}/p\mathbb{Z}$ but not both.

Note that a more general definition of Paley graphs extends this construction
to graphs over $q=p^n$ vertices, by using the finite field $F_q$ instead of
$\mathbb{Z}/p\mathbb{Z}$.
This construction requires to compute squares in general finite fields and is
not what is implemented here (i.e `paley_graph(25)` does not return the true
Paley graph associated with $5^2$).
parameters:


usage:
output_variable = networkx.generators.expanders.paley_graph()

**Maybe regular expander graph:**
Utility for creating a random regular expander.

Returns a random $d$-regular graph on $n$ nodes which is an expander
graph with very good probability.
parameters:
  - n: <class 'int'> = None -
  - d: <class 'int'> = None -
  - max_tries: <class 'int'> = 100 -
  - seed: int | None = None -

usage:
output_variable = networkx.generators.expanders.maybe_regular_expander_graph(n=<n_value>, d=<d_value>, max_tries=<max_tries_value>, seed=<seed_value>)

**Is regular expander:**
Determines whether the graph G is a regular expander. [1]_

An expander graph is a sparse graph with strong connectivity properties.

More precisely, this helper checks whether the graph is a
regular $(n, d, \lambda)$-expander with $\lambda$ close to
the Alon-Boppana bound and given by
$\lambda = 2 \sqrt{d - 1} + \epsilon$. [2]_

In the case where $\epsilon = 0$ then if the graph successfully passes the test
it is a Ramanujan graph. [3]_

A Ramanujan graph has spectral gap almost as large as possible, which makes them
excellent expanders.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.generators.expanders.is_regular_expander(G=<G_variable>)

**Random regular expander graph:**
Returns a random regular expander graph on $n$ nodes with degree $d$.

An expander graph is a sparse graph with strong connectivity properties. [1]_

More precisely the returned graph is a $(n, d, \lambda)$-expander with
$\lambda = 2 \sqrt{d - 1} + \epsilon$, close to the Alon-Boppana bound. [2]_

In the case where $\epsilon = 0$ it returns a Ramanujan graph.
A Ramanujan graph has spectral gap almost as large as possible,
which makes them excellent expanders. [3]_
parameters:
  - n: <class 'int'> = None -
  - d: <class 'int'> = None -
  - seed: int | None = None -

usage:
output_variable = networkx.generators.expanders.random_regular_expander_graph(n=<n_value>, d=<d_value>, seed=<seed_value>)
