**Is AT-free:**
Check if a graph is AT-free.

The method uses the `find_asteroidal_triple` method to recognize
an AT-free graph. If no asteroidal triple is found the graph is
AT-free and True is returned. If at least one asteroidal triple is
found the graph is not AT-free and False is returned.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --The graph to check whether is AT-free or not.
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.asteroidal.is_at_free(G=<G_variable>)
