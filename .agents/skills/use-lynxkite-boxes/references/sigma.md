**Sigma:**
Returns the small-world coefficient (sigma) of the given graph.

The small-world coefficient is defined as:
sigma = C/Cr / L/Lr
where C and L are respectively the average clustering coefficient and
average shortest path length of G. Cr and Lr are respectively the average
clustering coefficient and average shortest path length of an equivalent
random graph.

A graph is commonly classified as small-world if sigma>1.
parameters:
  - niter: <class 'int'> = 100 --Approximate number of rewiring per edge to compute the equivalent
random graph.
  - nrand: <class 'int'> = 10 --Number of random graphs generated to compute the average clustering
coefficient (Cr) and average shortest path length (Lr).
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.
  - G: <class 'networkx.classes.graph.Graph'> = ? --An undirected graph.
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.smallworld.sigma(niter=<niter_value>, nrand=<nrand_value>, seed=<seed_value>, G=<G_variable>)
