**Omega:**
Returns the small-world coefficient (omega) of a graph

The small-world coefficient of a graph G is:

omega = Lr/L - C/Cl

where C and L are respectively the average clustering coefficient and
average shortest path length of G. Lr is the average shortest path length
of an equivalent random graph and Cl is the average clustering coefficient
of an equivalent lattice graph.

The small-world coefficient (omega) measures how much G is like a lattice
or a random graph. Negative values mean G is similar to a lattice whereas
positive values mean G is a random graph.
Values close to 0 mean that G has small-world characteristics.
parameters:
  - niter: <class 'int'> = 5 --Approximate number of rewiring per edge to compute the equivalent
random graph.
  - nrand: <class 'int'> = 10 --Number of random graphs generated to compute the maximal clustering
coefficient (Cr) and average shortest path length (Lr).
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.
  - G: <class 'networkx.classes.graph.Graph'> = ? --An undirected graph.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.smallworld.omega(niter=<niter_value>, nrand=<nrand_value>, seed=<seed_value>, G=<G_variable>)
