**Full r-ary tree:**
Creates a full r-ary tree of `n` nodes.

Sometimes called a k-ary, n-ary, or m-ary tree.
"... all non-leaf nodes have exactly r children and all levels
are full except for some rightmost position of the bottom level
(if a leaf at the bottom level is missing, then so are all of the
leaves to its right." [1]_

.. plot::

    >>> nx.draw(nx.full_rary_tree(2, 10))
parameters:
  - r: <class 'int'> = ? --branching factor of the tree
  - n: <class 'int'> = ? --Number of nodes in the tree
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.generators.classic.full_rary_tree(r=<r_value>, n=<n_value>)
