**Planted partition graph:**
Returns the planted l-partition graph.

This model partitions a graph with n=l*k vertices in
l groups with k vertices each. Vertices of the same
group are linked with a probability p_in, and vertices
of different groups are linked with probability p_out.
parameters:
  - l: <class 'int'> = ? --Number of groups
  - k: <class 'int'> = ? --Number of vertices in each group
  - p_in: <class 'float'> = ? --probability of connecting vertices within a group
  - p_out: <class 'float'> = ? --probability of connected vertices between groups
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.generators.community.planted_partition_graph(l=<l_value>, k=<k_value>, p_in=<p_in_value>, p_out=<p_out_value>, seed=<seed_value>)
