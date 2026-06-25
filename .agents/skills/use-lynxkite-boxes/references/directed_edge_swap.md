**Directed edge swap:**
Swap three edges in a directed graph while keeping the node degrees fixed.

A directed edge swap swaps three edges such that a -> b -> c -> d becomes
a -> c -> b -> d. This pattern of swapping allows all possible states with the
same in- and out-degree distribution in a directed graph to be reached.

If the swap would create parallel edges (e.g. if a -> c already existed in the
previous example), another attempt is made to find a suitable trio of edges.
parameters:
  - nswap: <class 'int'> = 1 --Number of three-edge (directed) swaps to perform
  - max_tries: <class 'int'> = 100 --Maximum number of attempts to swap edges
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.
  - G: <class 'networkx.classes.graph.Graph'> = ? --A directed graph
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.swap.directed_edge_swap(nswap=<nswap_value>, max_tries=<max_tries_value>, seed=<seed_value>, G=<G_variable>)
