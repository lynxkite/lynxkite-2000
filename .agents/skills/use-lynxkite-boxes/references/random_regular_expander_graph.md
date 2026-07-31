**Random regular expander graph:**
Returns a random regular expander graph on $n$ nodes with degree $d$.

An expander graph is a sparse graph with strong connectivity properties. [1]_

More precisely the returned graph is a $(n, d, \lambda)$-expander with
$\lambda = 2 \sqrt{d - 1} + \epsilon$, close to the Alon-Boppana bound. [2]_

In the case where $\epsilon = 0$ it returns a Ramanujan graph.
A Ramanujan graph has spectral gap almost as large as possible,
which makes them excellent expanders. [3]_
parameters:
  - n: <class 'int'> = ? --The number of nodes.
  - d: <class 'int'> = ? --The degree of each node.
  - seed: int | None = ? --Seed used to set random number generation state. See :ref`Randomness<randomness>`.
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.generators.expanders.random_regular_expander_graph(n=<n_value>, d=<d_value>, seed=<seed_value>)
