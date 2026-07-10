**Identified nodes:**
Returns the graph that results from contracting `u` and `v`.

Node contraction identifies the two nodes as a single node incident to any
edge that was incident to the original two nodes.
Information about the contracted nodes and any modified edges are stored on
the output graph in a ``"contraction"`` attribute - see Examples for details.
parameters:
  - self_loops: <class 'bool'> = ? --If this is True, any edges joining `u` and `v` in `G` become
self-loops on the new node in the returned graph.
  - copy: <class 'bool'> = ? --If this is True (the default), make a copy of
`G` and return that instead of directly changing `G`.
  - G: <class 'networkx.classes.graph.Graph'> = ? --The graph whose nodes will be contracted.
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.minors.contraction.contracted_nodes(self_loops=<self_loops_value>, copy=<copy_value>, G=<G_variable>)
