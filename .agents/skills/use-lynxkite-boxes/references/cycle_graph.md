**Cycle graph:**
Returns the cycle graph $C_n$ of cyclically connected nodes.

$C_n$ is a path with its two end-nodes connected.

.. plot::

    >>> nx.draw(nx.cycle_graph(5))
parameters:
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.generators.classic.cycle_graph()
