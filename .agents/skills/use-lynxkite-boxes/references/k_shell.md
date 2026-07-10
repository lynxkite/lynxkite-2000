**k-shell:**
Returns the k-shell of G.

The k-shell is the subgraph induced by nodes with core number k.
That is, nodes in the k-core that are not in the (k+1)-core.
parameters:
  - k: int | None = ? --The order of the shell. If not specified return the outer shell.
  - G: <class 'networkx.classes.graph.Graph'> = ? --A graph or directed graph.
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.core.k_shell(k=<k_value>, G=<G_variable>)
