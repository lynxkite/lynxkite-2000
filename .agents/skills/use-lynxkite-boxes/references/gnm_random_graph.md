**G(n,m) random graph:**
Returns a $G_{n,m}$ random graph.

In the $G_{n,m}$ model, a graph is chosen uniformly at random from the set
of all graphs with $n$ nodes and $m$ edges.

This algorithm should be faster than :func:`dense_gnm_random_graph` for
sparse graphs.
parameters:
  - n: <class 'int'> = ? --The number of nodes.
  - m: <class 'int'> = ? --The number of edges.
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.
  - directed: bool | None = ? --If True return a directed graph
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.generators.random_graphs.gnm_random_graph(n=<n_value>, m=<m_value>, seed=<seed_value>, directed=<directed_value>)
