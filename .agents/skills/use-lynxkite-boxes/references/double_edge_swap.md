**Double edge swap:**
Swap two edges in the graph while keeping the node degrees fixed.

A double-edge swap removes two randomly chosen edges u-v and x-y
and creates the new edges u-x and v-y::

 u--v            u  v
        becomes  |  |
 x--y            x  y

If either the edge u-x or v-y already exist no swap is performed
and another attempt is made to find a suitable edge pair.
parameters:
  - nswap: <class 'int'> = 1 --Number of double-edge swaps to perform
  - max_tries: <class 'int'> = 100 --Maximum number of attempts to swap edges
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.
  - G: <class 'networkx.classes.graph.Graph'> = ? --An undirected graph

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.swap.double_edge_swap(nswap=<nswap_value>, max_tries=<max_tries_value>, seed=<seed_value>, G=<G_variable>)
