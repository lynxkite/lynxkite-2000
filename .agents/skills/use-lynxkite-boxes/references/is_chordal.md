**Is chordal:**
Checks whether G is a chordal graph.

A graph is chordal if every cycle of length at least 4 has a chord
(an edge joining two nodes not adjacent in the cycle).
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --A NetworkX graph.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.chordal.is_chordal(G=<G_variable>)
