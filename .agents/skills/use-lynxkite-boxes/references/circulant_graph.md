**Circulant graph:**
Returns the circulant graph $Ci_n(x_1, x_2, ..., x_m)$ with $n$ nodes.

The circulant graph $Ci_n(x_1, ..., x_m)$ consists of $n$ nodes $0, ..., n-1$
such that node $i$ is connected to nodes $(i + x) \mod n$ and $(i - x) \mod n$
for all $x$ in $x_1, ..., x_m$. Thus $Ci_n(1)$ is a cycle graph.

.. plot::

    >>> nx.draw(nx.circulant_graph(10, [1]))
parameters:
  - n: <class 'int'> = ? --The number of nodes in the graph.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.classic.circulant_graph(n=<n_value>)
