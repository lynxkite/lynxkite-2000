**To Prüfer sequence:**
Returns the Prüfer sequence of the given tree.

A *Prüfer sequence* is a list of *n* - 2 numbers between 0 and
*n* - 1, inclusive. The tree corresponding to a given Prüfer
sequence can be recovered by repeatedly joining a node in the
sequence with a node with the smallest potential degree according to
the sequence.
parameters:
  - T: <class 'networkx.classes.graph.Graph'> = ? --An undirected graph object representing a tree.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.tree.coding.to_prufer_sequence(T=<T_variable>)
