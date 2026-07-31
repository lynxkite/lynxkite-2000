**Tadpole graph:**
Returns the (m,n)-tadpole graph; ``C_m`` connected to ``P_n``.

This graph on m+n nodes connects a cycle of size `m` to a path of length `n`.
It looks like a tadpole. It is also called a kite graph or a dragon graph.

.. plot::

    >>> nx.draw(nx.tadpole_graph(3, 5))
parameters:
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.generators.classic.tadpole_graph()
