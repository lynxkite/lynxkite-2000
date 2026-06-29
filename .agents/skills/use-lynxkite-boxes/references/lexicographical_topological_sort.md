**Lexicographical topological sort:**
Generate the nodes in the unique lexicographical topological sort order.

Generates a unique ordering of nodes by first sorting topologically (for which there are often
multiple valid orderings) and then additionally by sorting lexicographically.

A topological sort arranges the nodes of a directed graph so that the
upstream node of each directed edge precedes the downstream node.
It is always possible to find a solution for directed graphs that have no cycles.
There may be more than one valid solution.

Lexicographical sorting is just sorting alphabetically. It is used here to break ties in the
topological sort and to determine a single, unique ordering.  This can be useful in comparing
sort results.

The lexicographical order can be customized by providing a function to the `key=` parameter.
The definition of the key function is the same as used in python's built-in `sort()`.
The function takes a single argument and returns a key to use for sorting purposes.

Lexicographical sorting can fail if the node names are un-sortable. See the example below.
The solution is to provide a function to the `key=` argument that returns sortable keys.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --A directed acyclic graph (DAG)

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.dag.lexicographical_topological_sort(G=<G_variable>)
