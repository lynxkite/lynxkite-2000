**Ladder graph:**
Returns the Ladder graph of length n.

This is two paths of n nodes, with
each pair connected by a single edge.

Node labels are the integers 0 to 2*n - 1.

.. plot::

    >>> nx.draw(nx.ladder_graph(5))
parameters:
  - n: <class 'int'> = ? --?

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.classic.ladder_graph(n=<n_value>)
