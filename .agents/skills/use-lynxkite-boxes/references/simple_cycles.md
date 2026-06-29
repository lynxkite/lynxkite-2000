**Simple cycles:**
Find simple cycles (elementary circuits) of a graph.

A "simple cycle", or "elementary circuit", is a closed path where
no node appears twice.  In a directed graph, two simple cycles are distinct
if they are not cyclic permutations of each other.  In an undirected graph,
two simple cycles are distinct if they are not cyclic permutations of each
other nor of the other's reversal.

Optionally, the cycles are bounded in length.  In the unbounded case, we use
a nonrecursive, iterator/generator version of Johnson's algorithm [1]_.  In
the bounded case, we use a version of the algorithm of Gupta and
Suzumura [2]_. There may be better algorithms for some cases [3]_ [4]_ [5]_.

The algorithms of Johnson, and Gupta and Suzumura, are enhanced by some
well-known preprocessing techniques.  When `G` is directed, we restrict our
attention to strongly connected components of `G`, generate all simple cycles
containing a certain node, remove that node, and further decompose the
remainder into strongly connected components.  When `G` is undirected, we
restrict our attention to biconnected components, generate all simple cycles
containing a particular edge, remove that edge, and further decompose the
remainder into biconnected components.

Note that multigraphs are supported by this function -- and in undirected
multigraphs, a pair of parallel edges is considered a cycle of length 2.
Likewise, self-loops are considered to be cycles of length 1.  We define
cycles as sequences of nodes; so the presence of loops and parallel edges
does not change the number of simple cycles in a graph.
parameters:
  - length_bound: int | None = ? --If `length_bound` is an int, generate all simple cycles of `G` with length at
most `length_bound`.  Otherwise, generate all simple cycles of `G`.
  - G: <class 'networkx.classes.graph.Graph'> = ? --A networkx graph. Undirected, directed, and multigraphs are all supported.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.cycles.simple_cycles(length_bound=<length_bound_value>, G=<G_variable>)
