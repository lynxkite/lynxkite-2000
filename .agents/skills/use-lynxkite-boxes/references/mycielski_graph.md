**Mycielski graph:**
Generator for the n_th Mycielski Graph.

The Mycielski family of graphs is an infinite set of graphs.
:math:`M_1` is the singleton graph, :math:`M_2` is two vertices with an
edge, and, for :math:`i > 2`, :math:`M_i` is the Mycielskian of
:math:`M_{i-1}`.

More information can be found at
http://mathworld.wolfram.com/MycielskiGraph.html
parameters:
  - n: <class 'int'> = ? --The desired Mycielski Graph.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.mycielski.mycielski_graph(n=<n_value>)
