**Chordless cycles:**
Find simple chordless cycles of a graph.

A `simple cycle` is a closed path where no node appears twice.  In a simple
cycle, a `chord` is an additional edge between two nodes in the cycle.  A
`chordless cycle` is a simple cycle without chords.  Said differently, a
chordless cycle is a cycle C in a graph G where the number of edges in the
induced graph G[C] is equal to the length of `C`.

Note that some care must be taken in the case that G is not a simple graph
nor a simple digraph.  Some authors limit the definition of chordless cycles
to have a prescribed minimum length; we do not.

    1. We interpret self-loops to be chordless cycles, except in multigraphs
       with multiple loops in parallel.  Likewise, in a chordless cycle of
       length greater than 1, there can be no nodes with self-loops.

    2. We interpret directed two-cycles to be chordless cycles, except in
       multi-digraphs when any edge in a two-cycle has a parallel copy.

    3. We interpret parallel pairs of undirected edges as two-cycles, except
       when a third (or more) parallel edge exists between the two nodes.

    4. Generalizing the above, edges with parallel clones may not occur in
       chordless cycles.

In a directed graph, two chordless cycles are distinct if they are not
cyclic permutations of each other.  In an undirected graph, two chordless
cycles are distinct if they are not cyclic permutations of each other nor of
the other's reversal.

Optionally, the cycles are bounded in length.

We use an algorithm strongly inspired by that of Dias et al [1]_.  It has
been modified in the following ways:

    1. Recursion is avoided, per Python's limitations.

    2. The labeling function is not necessary, because the starting paths
       are chosen (and deleted from the host graph) to prevent multiple
       occurrences of the same path.

    3. The search is optionally bounded at a specified length.

    4. Support for directed graphs is provided by extending cycles along
       forward edges, and blocking nodes along forward and reverse edges.

    5. Support for multigraphs is provided by omitting digons from the set
       of forward edges.
parameters:
  - length_bound: int | None = ? --If length_bound is an int, generate all simple cycles of G with length at
most length_bound.  Otherwise, generate all simple cycles of G.
  - G: <class 'networkx.classes.graph.Graph'> = ? --A directed graph

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.cycles.chordless_cycles(length_bound=<length_bound_value>, G=<G_variable>)
