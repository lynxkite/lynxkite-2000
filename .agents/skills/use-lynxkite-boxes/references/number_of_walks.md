**Number of walks:**
Returns the number of walks connecting each pair of nodes in `G`

A *walk* is a sequence of nodes in which each adjacent pair of nodes
in the sequence is adjacent in the graph. A walk can repeat the same
edge and go in the opposite direction just as people can walk on a
set of paths, but standing still is not counted as part of the walk.

This function only counts the walks with `walk_length` edges. Note that
the number of nodes in the walk sequence is one more than `walk_length`.
The number of walks can grow very quickly on a larger graph
and with a larger walk length.
parameters:
  - walk_length: <class 'int'> = ? --A nonnegative integer representing the length of a walk.
  - G: <class 'networkx.classes.graph.Graph'> = ? --?
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.walks.number_of_walks(walk_length=<walk_length_value>, G=<G_variable>)
