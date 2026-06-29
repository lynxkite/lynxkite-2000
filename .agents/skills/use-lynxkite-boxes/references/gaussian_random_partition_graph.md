**Gaussian random partition graph:**
Generate a Gaussian random partition graph.

A Gaussian random partition graph is created by creating k partitions
each with a size drawn from a normal distribution with mean s and variance
s/v. Nodes are connected within clusters with probability p_in and
between clusters with probability p_out[1]
parameters:
  - n: <class 'int'> = ? --Number of nodes in the graph
  - s: <class 'float'> = ? --Mean cluster size
  - v: <class 'float'> = ? --Shape parameter. The variance of cluster size distribution is s/v.
  - p_in: <class 'float'> = ? --Probability of intra cluster connection.
  - p_out: <class 'float'> = ? --Probability of inter cluster connection.
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.community.gaussian_random_partition_graph(n=<n_value>, s=<s_value>, v=<v_value>, p_in=<p_in_value>, p_out=<p_out_value>, seed=<seed_value>)
