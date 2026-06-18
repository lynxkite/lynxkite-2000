---
name: networkx-algorithms-cycles
description: Collection of operations - Cycle basis, Simple cycles, Recursive simple cycles, Find cycle, Minimum cycle basis, Chordless cycles, Girth
---

**Cycle basis:**
Returns a list of cycles which form a basis for cycles of G.

A basis for cycles of a network is a minimal collection of
cycles such that any cycle in the network can be written
as a sum of cycles in the basis.  Here summation of cycles
is defined as "exclusive or" of the edges. Cycle bases are
useful, e.g. when deriving equations for electric circuits
using Kirchhoff's Laws.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.algorithms.cycles.cycle_basis(G=<G_variable>)

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
  - length_bound: int | None = None - .
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.algorithms.cycles.simple_cycles(length_bound=<length_bound_value>, G=<G_variable>)

**Recursive simple cycles:**
Find simple cycles (elementary circuits) of a directed graph.

A `simple cycle`, or `elementary circuit`, is a closed path where
no node appears twice. Two elementary circuits are distinct if they
are not cyclic permutations of each other.

This version uses a recursive algorithm to build a list of cycles.
You should probably use the iterator version called simple_cycles().
Warning: This recursive version uses lots of RAM!
It appears in NetworkX for pedagogical value.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.algorithms.cycles.recursive_simple_cycles(G=<G_variable>)

**Find cycle:**
Returns a cycle found via depth-first traversal.

The cycle is a list of edges indicating the cyclic path.
Orientation of directed edges is controlled by `orientation`.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.algorithms.cycles.find_cycle(G=<G_variable>)

**Minimum cycle basis:**
Returns a minimum weight cycle basis for G

Minimum weight means a cycle basis for which the total weight
(length for unweighted graphs) of all the cycles is minimum.
parameters:
  - weight: <class 'str'> = None - .
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.algorithms.cycles.minimum_cycle_basis(weight=<weight_value>, G=<G_variable>)

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
  - length_bound: int | None = None - .
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.algorithms.cycles.chordless_cycles(length_bound=<length_bound_value>, G=<G_variable>)

**Girth:**
Returns the girth of the graph.

The girth of a graph is the length of its shortest cycle, or infinity if
the graph is acyclic. The algorithm follows the description given on the
Wikipedia page [1]_, and runs in time O(mn) on a graph with m edges and n
nodes.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.algorithms.cycles.girth(G=<G_variable>)
