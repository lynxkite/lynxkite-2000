**Shortest simple paths:**
Returns
-------
path_generator: generator
   A generator that produces lists of simple paths, in order from
   shortest to longest.

Raises
------
NetworkXNoPath
   If no path exists between source and target.

NetworkXError
   If source or target nodes are not in the input graph.

NetworkXNotImplemented
   If the input graph is a Multi[Di]Graph.

Examples
--------

>>> G = nx.cycle_graph(7)
>>> paths = list(nx.shortest_simple_paths(G, 0, 3))
>>> print(paths)
[[0, 1, 2, 3], [0, 6, 5, 4, 3]]

You can use this function to efficiently compute the k shortest/best
paths between two nodes.

>>> from itertools import islice
>>> def k_shortest_paths(G, source, target, k, weight=None):
...     return list(
...         islice(nx.shortest_simple_paths(G, source, target, weight=weight), k)
...     )
>>> for path in k_shortest_paths(G, 0, 3, 2):
...     print(path)
[0, 1, 2, 3]
[0, 6, 5, 4, 3]

Notes
-----
This procedure is based on algorithm by Jin Y. Yen [1]_.  Finding
the first $K$ paths requires $O(KN^3)$ operations.

See Also
--------
all_shortest_paths
shortest_path
all_simple_paths

References
----------
.. [1] Jin Y. Yen, "Finding the K Shortest Loopless Paths in a
   Network", Management Science, Vol. 17, No. 11, Theory Series
   (Jul., 1971), pp. 712-716.
parameters:
  - weight: <class 'str'> = ? --?
  - G: <class 'networkx.classes.graph.Graph'> = ? --?
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.simple_paths.shortest_simple_paths(weight=<weight_value>, G=<G_variable>)
