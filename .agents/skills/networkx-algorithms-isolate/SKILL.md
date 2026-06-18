---
name: networkx-algorithms-isolate
description: Collection of operations - Isolates, Number of isolates
---

**Isolates:**
Iterator over isolates in the graph.

An *isolate* is a node with no neighbors (that is, with degree
zero). For directed graphs, this means no in-neighbors and no
out-neighbors.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.isolate.isolates(G=<G_variable>)

**Number of isolates:**
Returns the number of isolates in the graph.

An *isolate* is a node with no neighbors (that is, with degree
zero). For directed graphs, this means no in-neighbors and no
out-neighbors.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.isolate.number_of_isolates(G=<G_variable>)
