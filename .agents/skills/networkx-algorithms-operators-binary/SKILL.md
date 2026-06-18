---
name: networkx-algorithms-operators-binary
description: Collection of operations - Union, Compose, Disjoint union, Difference, Symmetric difference, Full join
---

**Union:**
Combine graphs G and H. The names of nodes must be unique.

A name collision between the graphs will raise an exception.

A renaming facility is provided to avoid name collisions.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = None -
  - H: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.operators.binary.union(G=<G_variable>, H=<H_variable>)

**Compose:**
Compose graph G with H by combining nodes and edges into a single graph.

The node sets and edges sets do not need to be disjoint.

Composing preserves the attributes of nodes and edges.
Attribute values from H take precedent over attribute values from G.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = None -
  - H: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.operators.binary.compose(G=<G_variable>, H=<H_variable>)

**Disjoint union:**
Combine graphs G and H. The nodes are assumed to be unique (disjoint).

This algorithm automatically relabels nodes to avoid name collisions.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = None -
  - H: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.operators.binary.disjoint_union(G=<G_variable>, H=<H_variable>)

**Difference:**
Returns a new graph that contains the edges that exist in G but not in H.

The node sets of H and G must be the same.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = None -
  - H: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.operators.binary.difference(G=<G_variable>, H=<H_variable>)

**Symmetric difference:**
Returns new graph with edges that exist in either G or H but not both.

The node sets of H and G must be the same.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = None -
  - H: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.operators.binary.symmetric_difference(G=<G_variable>, H=<H_variable>)

**Full join:**
Returns the full join of graphs G and H.

Full join is the union of G and H in which all edges between
G and H are added.
The node sets of G and H must be disjoint,
otherwise an exception is raised.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = None -
  - H: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.operators.binary.full_join(G=<G_variable>, H=<H_variable>)
