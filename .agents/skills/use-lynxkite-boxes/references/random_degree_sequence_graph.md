**Random degree sequence graph:**
Returns a simple random graph with the given degree sequence.

If the maximum degree $d_m$ in the sequence is $O(m^{1/4})$ then the
algorithm produces almost uniform random graphs in $O(m d_m)$ time
where $m$ is the number of edges.
parameters:
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.
  - tries: int | None = 10 --Maximum number of tries to create a graph
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.generators.degree_seq.random_degree_sequence_graph(seed=<seed_value>, tries=<tries_value>)
