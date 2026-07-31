**Relaxed caveman graph:**
Returns a relaxed caveman graph.

A relaxed caveman graph starts with `l` cliques of size `k`.  Edges are
then randomly rewired with probability `p` to link different cliques.
parameters:
  - l: <class 'int'> = ? --Number of groups
  - k: <class 'int'> = ? --Size of cliques
  - p: <class 'float'> = ? --Probability of rewiring each edge.
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.generators.community.relaxed_caveman_graph(l=<l_value>, k=<k_value>, p=<p_value>, seed=<seed_value>)
