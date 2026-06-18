---
name: networkx-algorithms-swap
description: Collection of operations - Double edge swap, Connected double edge swap, Directed edge swap
---

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
  - nswap: <class 'int'> = 1 - .
  - max_tries: <class 'int'> = 100 - .
  - seed: int | None = None - .
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.algorithms.swap.double_edge_swap(nswap=<nswap_value>, max_tries=<max_tries_value>, seed=<seed_value>, G=<G_variable>)

**Connected double edge swap:**
Attempts the specified number of double-edge swaps in the graph `G`.

A double-edge swap removes two randomly chosen edges `(u, v)` and `(x,
y)` and creates the new edges `(u, x)` and `(v, y)`::

 u--v            u  v
        becomes  |  |
 x--y            x  y

If either `(u, x)` or `(v, y)` already exist, then no swap is performed
so the actual number of swapped edges is always *at most* `nswap`.
parameters:
  - nswap: <class 'int'> = 1 - .
  - _window_threshold: <class 'int'> = 3 - .
  - seed: int | None = None - .
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.algorithms.swap.connected_double_edge_swap(nswap=<nswap_value>, _window_threshold=<_window_threshold_value>, seed=<seed_value>, G=<G_variable>)

**Directed edge swap:**
Swap three edges in a directed graph while keeping the node degrees fixed.

A directed edge swap swaps three edges such that a -> b -> c -> d becomes
a -> c -> b -> d. This pattern of swapping allows all possible states with the
same in- and out-degree distribution in a directed graph to be reached.

If the swap would create parallel edges (e.g. if a -> c already existed in the
previous example), another attempt is made to find a suitable trio of edges.
parameters:
  - nswap: <class 'int'> = 1 - .
  - max_tries: <class 'int'> = 100 - .
  - seed: int | None = None - .
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.algorithms.swap.directed_edge_swap(nswap=<nswap_value>, max_tries=<max_tries_value>, seed=<seed_value>, G=<G_variable>)
