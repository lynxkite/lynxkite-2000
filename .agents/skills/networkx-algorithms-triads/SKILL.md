---
name: networkx-algorithms-triads
description: Collection of operations - Is triad, All triads, Triads by type, Triad type
---

**Is triad:**
Returns True if the graph G is a triad, else False.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.algorithms.triads.is_triad(G=<G_variable>)

**All triads:**
A generator of all possible triads in G.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.algorithms.triads.all_triads(G=<G_variable>)

**Triads by type:**
Returns a list of all triads for each triad type in a directed graph.
There are exactly 16 different types of triads possible. Suppose 1, 2, 3 are three
nodes, they will be classified as a particular triad type if their connections
are as follows:

- 003: 1, 2, 3
- 012: 1 -> 2, 3
- 102: 1 <-> 2, 3
- 021D: 1 <- 2 -> 3
- 021U: 1 -> 2 <- 3
- 021C: 1 -> 2 -> 3
- 111D: 1 <-> 2 <- 3
- 111U: 1 <-> 2 -> 3
- 030T: 1 -> 2 -> 3, 1 -> 3
- 030C: 1 <- 2 <- 3, 1 -> 3
- 201: 1 <-> 2 <-> 3
- 120D: 1 <- 2 -> 3, 1 <-> 3
- 120U: 1 -> 2 <- 3, 1 <-> 3
- 120C: 1 -> 2 -> 3, 1 <-> 3
- 210: 1 -> 2 <-> 3, 1 <-> 3
- 300: 1 <-> 2 <-> 3, 1 <-> 3

Refer to the :doc:`example gallery </auto_examples/graph/plot_triad_types>`
for visual examples of the triad types.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.algorithms.triads.triads_by_type(G=<G_variable>)

**Triad type:**
Returns the sociological triad type for a triad.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.algorithms.triads.triad_type(G=<G_variable>)
