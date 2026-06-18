---
name: networkx-linalg-algebraicconnectivity
description: Collection of operations - Algebraic connectivity, Fiedler vector, Spectral ordering, Spectral bisection
---

**Algebraic connectivity:**
Returns the algebraic connectivity of an undirected graph.

The algebraic connectivity of a connected undirected graph is the second
smallest eigenvalue of its Laplacian matrix.
parameters:
  - weight: str | None = weight -
  - normalized: bool | None = None -
  - tol: float | None = 1e-08 -
  - method: str | None = tracemin_pcg -
  - seed: int | None = None -
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.linalg.algebraicconnectivity.algebraic_connectivity(weight=<weight_value>, normalized=<normalized_value>, tol=<tol_value>, method=<method_value>, seed=<seed_value>, G=<G_variable>)

**Fiedler vector:**
Returns the Fiedler vector of a connected undirected graph.

The Fiedler vector of a connected undirected graph is the eigenvector
corresponding to the second smallest eigenvalue of the Laplacian matrix
of the graph.
parameters:
  - weight: str | None = weight -
  - normalized: bool | None = None -
  - tol: float | None = 1e-08 -
  - method: str | None = tracemin_pcg -
  - seed: int | None = None -
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.linalg.algebraicconnectivity.fiedler_vector(weight=<weight_value>, normalized=<normalized_value>, tol=<tol_value>, method=<method_value>, seed=<seed_value>, G=<G_variable>)

**Spectral ordering:**
Compute the spectral_ordering of a graph.

The spectral ordering of a graph is an ordering of its nodes where nodes
in the same weakly connected components appear contiguous and ordered by
their corresponding elements in the Fiedler vector of the component.
parameters:
  - weight: str | None = weight -
  - normalized: bool | None = None -
  - tol: float | None = 1e-08 -
  - method: str | None = tracemin_pcg -
  - seed: int | None = None -
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.linalg.algebraicconnectivity.spectral_ordering(weight=<weight_value>, normalized=<normalized_value>, tol=<tol_value>, method=<method_value>, seed=<seed_value>, G=<G_variable>)

**Spectral bisection:**
Bisect the graph using the Fiedler vector.

This method uses the Fiedler vector to bisect a graph.
The partition is defined by the nodes which are associated with
either positive or negative values in the vector.
parameters:
  - weight: str | None = weight -
  - normalized: bool | None = None -
  - tol: float | None = 1e-08 -
  - method: str | None = tracemin_pcg -
  - seed: int | None = None -
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.linalg.algebraicconnectivity.spectral_bisection(weight=<weight_value>, normalized=<normalized_value>, tol=<tol_value>, method=<method_value>, seed=<seed_value>, G=<G_variable>)
