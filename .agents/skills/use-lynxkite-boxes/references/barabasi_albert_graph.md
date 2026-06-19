**Barabasi–Albert graph:**
Returns a random graph using Barabási–Albert preferential attachment

A graph of $n$ nodes is grown by attaching new nodes each with $m$
edges that are preferentially attached to existing nodes with high degree.
parameters:
  - n: <class 'int'> = ? --Number of nodes
  - m: <class 'int'> = ? --Number of edges to attach from a new node to existing nodes
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.random_graphs.barabasi_albert_graph(n=<n_value>, m=<m_value>, seed=<seed_value>)
