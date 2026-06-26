**DAG to branching:**
Returns a branching representing all (overlapping) paths from
root nodes to leaf nodes in the given directed acyclic graph.

As described in :mod:`networkx.algorithms.tree.recognition`, a
*branching* is a directed forest in which each node has at most one
parent. In other words, a branching is a disjoint union of
*arborescences*. For this function, each node of in-degree zero in
`G` becomes a root of one of the arborescences, and there will be
one leaf node for each distinct path from that root to a leaf node
in `G`.

Each node `v` in `G` with *k* parents becomes *k* distinct nodes in
the returned branching, one for each parent, and the sub-DAG rooted
at `v` is duplicated for each copy. The algorithm then recurses on
the children of each copy of `v`.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --A directed acyclic graph.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.dag.dag_to_branching(G=<G_variable>)
