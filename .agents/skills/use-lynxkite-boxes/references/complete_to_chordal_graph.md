**Complete to chordal graph:**
Return a copy of G completed to a chordal graph

Adds edges to a copy of G to create a chordal graph. A graph G=(V,E) is
called chordal if for each cycle with length bigger than 3, there exist
two non-adjacent nodes connected by an edge (called a chord).
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --Undirected graph

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.chordal.complete_to_chordal_graph(G=<G_variable>)
