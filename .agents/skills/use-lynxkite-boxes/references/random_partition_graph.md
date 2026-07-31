**Random partition graph:**
Returns the random partition graph with a partition of sizes.

A partition graph is a graph of communities with sizes defined by
s in sizes. Nodes in the same group are connected with probability
p_in and nodes of different groups are connected with probability
p_out.
parameters:
  - p_in: <class 'float'> = ? --probability of edges with in groups
  - p_out: <class 'float'> = ? --probability of edges between groups
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.generators.community.random_partition_graph(p_in=<p_in_value>, p_out=<p_out_value>, seed=<seed_value>)
