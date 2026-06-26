**Circular ladder graph:**
Returns the circular ladder graph $CL_n$ of length n.

$CL_n$ consists of two concentric n-cycles in which
each of the n pairs of concentric nodes are joined by an edge.

Node labels are the integers 0 to n-1

.. plot::

    >>> nx.draw(nx.circular_ladder_graph(5))
parameters:
  - n: <class 'int'> = ? --?

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.classic.circular_ladder_graph(n=<n_value>)
