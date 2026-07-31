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
  - nswap: <class 'int'> = 1 --Number of double-edge swaps to perform
  - _window_threshold: <class 'int'> = 3 --
The window size below which connectedness of the graph will be checked
after each swap.

The "window" in this function is a dynamically updated integer that
represents the number of swap attempts to make before checking if the
graph remains connected. It is an optimization used to decrease the
running time of the algorithm in exchange for increased complexity of
implementation.

If the window size is below this threshold, then the algorithm checks
after each swap if the graph remains connected by checking if there is a
path joining the two nodes whose edge was just removed. If the window
size is above this threshold, then the algorithm performs do all the
swaps in the window and only then check if the graph is still connected.
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.
  - G: <class 'networkx.classes.graph.Graph'> = ? --An undirected graph
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.swap.connected_double_edge_swap(nswap=<nswap_value>, _window_threshold=<_window_threshold_value>, seed=<seed_value>, G=<G_variable>)
