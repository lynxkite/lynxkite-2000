**Binomial tree:**
Returns the Binomial Tree of order n.

The binomial tree of order 0 consists of a single node. A binomial tree of order k
is defined recursively by linking two binomial trees of order k-1: the root of one is
the leftmost child of the root of the other.

.. plot::

    >>> nx.draw(nx.binomial_tree(3))
parameters:
  - n: <class 'int'> = ? --Order of the binomial tree.
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.generators.classic.binomial_tree(n=<n_value>)
