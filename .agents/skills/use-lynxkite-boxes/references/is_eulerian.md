**Is Eulerian:**
Returns True if and only if `G` is Eulerian.

A graph is *Eulerian* if it has an Eulerian circuit. An *Eulerian
circuit* is a closed walk that includes each edge of a graph exactly
once.

Graphs with isolated vertices (i.e. vertices with zero degree) are not
considered to have Eulerian circuits. Therefore, if the graph is not
connected (or not strongly connected, for directed graphs), this function
returns False.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --A graph, either directed or undirected.
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.euler.is_eulerian(G=<G_variable>)
