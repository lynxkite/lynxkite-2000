---
name: networkx-linalg-graphmatrix
description: Collection of operations - Incidence matrix, Adjacency matrix
---

**Incidence matrix:**
Returns incidence matrix of G.

The incidence matrix assigns each row to a node and each column to an edge.
For a standard incidence matrix a 1 appears wherever a row's node is
incident on the column's edge.  For an oriented incidence matrix each
edge is assigned an orientation (arbitrarily for undirected and aligning to
direction for directed).  A -1 appears for the source (tail) of an edge and
1 for the destination (head) of the edge.  The elements are zero otherwise.
parameters:
  - oriented: bool | None = None - .
  - weight: str | None = None - .
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.linalg.graphmatrix.incidence_matrix(oriented=<oriented_value>, weight=<weight_value>, G=<G_variable>)

**Adjacency matrix:**
Returns adjacency matrix of `G`.
parameters:
  - weight: str | None = weight - .
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.linalg.graphmatrix.adjacency_matrix(weight=<weight_value>, G=<G_variable>)
