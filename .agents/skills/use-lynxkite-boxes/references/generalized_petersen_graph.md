**Generalized petersen graph:**
Returns the Generalized Petersen Graph GP(n,k).

The Generalized Peterson Graph consists of an outer cycle of n nodes
connected to an inner circulant graph of n nodes, where nodes in the
inner circulant are connected to their kth nearest neighbor [1]_ [2]_.
A Generalized Petersen Graph is cubic with 2n nodes and 3n edges.

Some well known graphs are examples of Generalized Petersen Graphs such
as the Petersen Graph GP(5, 2), the Desargues graph GP(10, 3), the
Moebius-Kantor graph GP(8, 3), and the dodecahedron graph GP(10, 2).
parameters:
  - n: <class 'int'> = ? --Number of nodes in the outer cycle and inner circulant. ``n >= 3`` is required.
  - k: <class 'int'> = ? --Neighbor to connect in the inner circulant. ``1 <= k <= n/2``.
Note that some people require ``k < n/2`` but we and others allow equality.
Also, ``k < n/2`` is equivalent to ``k <= floor((n-1)/2)``

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.small.generalized_petersen_graph(n=<n_value>, k=<k_value>)
