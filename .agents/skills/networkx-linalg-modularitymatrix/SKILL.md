---
name: networkx-linalg-modularitymatrix
description: Collection of operations - Modularity matrix, Directed modularity matrix
---

**Modularity matrix:**
Returns the modularity matrix of G.

The modularity matrix is the matrix B = A - <A>, where A is the adjacency
matrix and <A> is the average adjacency matrix, assuming that the graph
is described by the configuration model.

More specifically, the element B_ij of B is defined as

.. math::
    A_{ij} - {k_i k_j \over 2 m}

where k_i is the degree of node i, and where m is the number of edges
in the graph. When weight is set to a name of an attribute edge, Aij, k_i,
k_j and m are computed using its value.
parameters:
  - weight: str | None = ? --The edge attribute that holds the numerical value used for
the edge weight.  If None then all edge weights are 1.
  - G: <class 'networkx.classes.graph.Graph'> = ? --A NetworkX graph

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.linalg.modularitymatrix.modularity_matrix(weight=<weight_value>, G=<G_variable>)

**Directed modularity matrix:**
Returns the directed modularity matrix of G.

The modularity matrix is the matrix B = A - <A>, where A is the adjacency
matrix and <A> is the expected adjacency matrix, assuming that the graph
is described by the configuration model.

More specifically, the element B_ij of B is defined as

.. math::
    B_{ij} = A_{ij} - k_i^{out} k_j^{in} / m

where :math:`k_i^{in}` is the in degree of node i, and :math:`k_j^{out}` is the out degree
of node j, with m the number of edges in the graph. When weight is set
to a name of an attribute edge, Aij, k_i, k_j and m are computed using
its value.
parameters:
  - weight: str | None = ? --The edge attribute that holds the numerical value used for
the edge weight.  If None then all edge weights are 1.
  - G: <class 'networkx.classes.graph.Graph'> = ? --A NetworkX DiGraph

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.linalg.modularitymatrix.directed_modularity_matrix(weight=<weight_value>, G=<G_variable>)
