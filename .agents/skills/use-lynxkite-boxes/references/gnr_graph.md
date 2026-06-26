**G(n,r) graph:**
Returns the growing network with redirection (GNR) digraph with `n`
nodes and redirection probability `p`.

The GNR graph is built by adding nodes one at a time with a link to one
previously added node.  The previous target node is chosen uniformly at
random.  With probability `p` the link is instead "redirected" to the
successor node of the target.

The graph is always a (directed) tree.
parameters:
  - n: <class 'int'> = ? --The number of nodes for the generated graph.
  - p: <class 'float'> = ? --The redirection probability.
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.directed.gnr_graph(n=<n_value>, p=<p_value>, seed=<seed_value>)
