**Turan graph:**
Return the Turan Graph

The Turan Graph is a complete multipartite graph on $n$ nodes
with $r$ disjoint subsets. That is, edges connect each node to
every node not in its subset.

Given $n$ and $r$, we create a complete multipartite graph with
$r-(n \mod r)$ partitions of size $n/r$, rounded down, and
$n \mod r$ partitions of size $n/r+1$, rounded down.

.. plot::

    >>> nx.draw(nx.turan_graph(6, 2))
parameters:
  - n: <class 'int'> = ? --The number of nodes.
  - r: <class 'int'> = ? --The number of partitions.
Must be less than or equal to n.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.classic.turan_graph(n=<n_value>, r=<r_value>)
