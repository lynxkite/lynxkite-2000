**Is branching:**
Returns True if `G` is a branching.

A branching is a directed forest with maximum in-degree equal to 1.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --The directed graph to test.
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.tree.recognition.is_branching(G=<G_variable>)
