---
name: networkx-algorithms-minors-contraction
description: Collection of operations - Contracted nodes, Identified nodes
---

**Contracted nodes:**
Returns the graph that results from contracting `u` and `v`.

Node contraction identifies the two nodes as a single node incident to any
edge that was incident to the original two nodes.
Information about the contracted nodes and any modified edges are stored on
the output graph in a ``"contraction"`` attribute - see Examples for details.
parameters:
  - self_loops: <class 'bool'> = None -
  - copy: <class 'bool'> = None -
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.minors.contraction.contracted_nodes(self_loops=<self_loops_value>, copy=<copy_value>, G=<G_variable>)

**Identified nodes:**
Returns the graph that results from contracting `u` and `v`.

Node contraction identifies the two nodes as a single node incident to any
edge that was incident to the original two nodes.
Information about the contracted nodes and any modified edges are stored on
the output graph in a ``"contraction"`` attribute - see Examples for details.
parameters:
  - self_loops: <class 'bool'> = None -
  - copy: <class 'bool'> = None -
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.minors.contraction.contracted_nodes(self_loops=<self_loops_value>, copy=<copy_value>, G=<G_variable>)
