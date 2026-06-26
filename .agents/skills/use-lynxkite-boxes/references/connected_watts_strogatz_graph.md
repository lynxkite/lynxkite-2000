**Connected Watts–Strogatz graph:**
Returns a connected Watts–Strogatz small-world graph.

Attempts to generate a connected graph by repeated generation of
Watts–Strogatz small-world graphs.  An exception is raised if the maximum
number of tries is exceeded.
parameters:
  - n: <class 'int'> = ? --The number of nodes
  - k: <class 'int'> = ? --Each node is joined with its `k` nearest neighbors in a ring
topology.
  - p: <class 'float'> = ? --The probability of rewiring each edge
  - tries: <class 'int'> = 100 --Number of attempts to generate a connected graph.
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.random_graphs.connected_watts_strogatz_graph(n=<n_value>, k=<k_value>, p=<p_value>, tries=<tries_value>, seed=<seed_value>)
