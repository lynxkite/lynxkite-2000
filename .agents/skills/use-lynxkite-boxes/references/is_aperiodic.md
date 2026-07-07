**Is aperiodic:**
Returns True if `G` is aperiodic.

A strongly connected directed graph is aperiodic if there is no integer ``k > 1``
that divides the length of every cycle in the graph.

This function requires the graph `G` to be strongly connected and will raise
an error if it's not. For graphs that are not strongly connected, you should
first identify their strongly connected components
(using :func:`~networkx.algorithms.components.strongly_connected_components`)
or attracting components
(using :func:`~networkx.algorithms.components.attracting_components`),
and then apply this function to those individual components.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --A directed graph
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.dag.is_aperiodic(G=<G_variable>)
