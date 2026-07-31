**Duplication divergence graph:**
Returns an undirected graph using the duplication-divergence model.

A graph of `n` nodes is created by duplicating the initial nodes
and retaining edges incident to the original nodes with a retention
probability `p`.
parameters:
  - n: <class 'int'> = ? --The desired number of nodes in the graph.
  - p: <class 'float'> = ? --The probability for retaining the edge of the replicated node.
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.generators.duplication.duplication_divergence_graph(n=<n_value>, p=<p_value>, seed=<seed_value>)
