**Barbell graph:**
Returns the Barbell Graph: two complete graphs connected by a path.

.. plot::

    >>> nx.draw(nx.barbell_graph(4, 2))
parameters:
  - m1: <class 'int'> = ? --Size of the left and right barbells, must be greater than 2.
  - m2: <class 'int'> = ? --Length of the path connecting the barbells.
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.generators.classic.barbell_graph(m1=<m1_value>, m2=<m2_value>)
