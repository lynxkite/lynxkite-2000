**Balanced tree:**
Returns the perfectly balanced `r`-ary tree of height `h`.

.. plot::

    >>> nx.draw(nx.balanced_tree(2, 3))
parameters:
  - r: <class 'int'> = ? --Branching factor of the tree; each node will have `r`
children.
  - h: <class 'int'> = ? --Height of the tree.
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.generators.classic.balanced_tree(r=<r_value>, h=<h_value>)
