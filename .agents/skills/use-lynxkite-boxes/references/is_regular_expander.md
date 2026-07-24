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
  - G: <class 'networkx.classes.graph.Graph'> = ? --?
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.generators.expanders.is_regular_expander(G=<G_variable>)
