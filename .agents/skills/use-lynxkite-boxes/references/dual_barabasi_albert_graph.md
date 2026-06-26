**Dual Barabasi–Albert graph:**
Returns a random graph using dual Barabási–Albert preferential attachment

A graph of $n$ nodes is grown by attaching new nodes each with either $m_1$
edges (with probability $p$) or $m_2$ edges (with probability $1-p$) that
are preferentially attached to existing nodes with high degree.
parameters:
  - n: <class 'int'> = ? --Number of nodes
  - m1: <class 'int'> = ? --Number of edges to link each new node to existing nodes with probability $p$
  - m2: <class 'int'> = ? --Number of edges to link each new node to existing nodes with probability $1-p$
  - p: <class 'float'> = ? --The probability of attaching $m_1$ edges (as opposed to $m_2$ edges)
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.random_graphs.dual_barabasi_albert_graph(n=<n_value>, m1=<m1_value>, m2=<m2_value>, p=<p_value>, seed=<seed_value>)
