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
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.generators.expanders.paley_graph()
