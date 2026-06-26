**Find minimal d-separator:**
Returns a minimal d-separating set between `x` and `y` if possible

A d-separating set in a DAG is a set of nodes that blocks all
paths between the two sets of nodes, `x` and `y`. This function
constructs a d-separating set that is "minimal", meaning no nodes can
be removed without it losing the d-separating property for `x` and `y`.
If no d-separating sets exist for `x` and `y`, this returns `None`.

In a DAG there may be more than one minimal d-separator between two
sets of nodes. Minimal d-separators are not always unique. This function
returns one minimal d-separator, or `None` if no d-separator exists.

Uses the algorithm presented in [1]_. The complexity of the algorithm
is :math:`O(m)`, where :math:`m` stands for the number of edges in
the subgraph of G consisting of only the ancestors of `x` and `y`.
For full details, see [1]_.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --A networkx DAG.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.d_separation.find_minimal_d_separator(G=<G_variable>)
