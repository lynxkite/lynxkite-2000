**Fiedler vector:**
Returns the Fiedler vector of a connected undirected graph.

The Fiedler vector of a connected undirected graph is the eigenvector
corresponding to the second smallest eigenvalue of the Laplacian matrix
of the graph.
parameters:
  - weight: str | None = weight --The data key used to determine the weight of each edge. If None, then
each edge has unit weight.
  - normalized: bool | None = ? --Whether the normalized Laplacian matrix is used.
  - tol: float | None = 1e-08 --Tolerance of relative residual in eigenvalue computation.
  - method: str | None = tracemin_pcg --Method of eigenvalue computation. It must be one of the tracemin
options shown below (TraceMIN), 'lanczos' (Lanczos iteration)
or 'lobpcg' (LOBPCG).

The TraceMIN algorithm uses a linear system solver. The following
values allow specifying the solver to be used.

=============== ========================================
Value           Solver
=============== ========================================
'tracemin_pcg'  Preconditioned conjugate gradient method
'tracemin_lu'   LU factorization
=============== ========================================
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.
  - G: <class 'networkx.classes.graph.Graph'> = ? --An undirected graph.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.linalg.algebraicconnectivity.fiedler_vector(weight=<weight_value>, normalized=<normalized_value>, tol=<tol_value>, method=<method_value>, seed=<seed_value>, G=<G_variable>)
