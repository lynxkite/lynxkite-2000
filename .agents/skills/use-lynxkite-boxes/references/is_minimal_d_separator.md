**Is minimal d-separator:**
Determine if `z` is a minimal d-separator for `x` and `y`.

A d-separator, `z`, in a DAG is a set of nodes that blocks
all paths from nodes in set `x` to nodes in set `y`.
A minimal d-separator is a d-separator `z` such that removing
any subset of nodes makes it no longer a d-separator.

Note: This function checks whether `z` is a d-separator AND is
minimal. One can use the function `is_d_separator` to only check if
`z` is a d-separator. See examples below.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --A NetworkX DAG.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.d_separation.is_minimal_d_separator(G=<G_variable>)
