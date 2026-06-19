---
name: networkx-algorithms-smallworld
description: Collection of operations - Random reference, Lattice reference, Sigma, Omega
---

**Random reference:**
Compute a random graph by swapping edges of a given graph.
parameters:
  - niter: <class 'int'> = 1 --An edge is rewired approximately `niter` times.
  - connectivity: <class 'bool'> = ? --When True, ensure connectivity for the randomized graph.
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.
  - G: <class 'networkx.classes.graph.Graph'> = ? --An undirected graph with 4 or more nodes.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.smallworld.random_reference(niter=<niter_value>, connectivity=<connectivity_value>, seed=<seed_value>, G=<G_variable>)

**Lattice reference:**
Latticize the given graph by swapping edges.
parameters:
  - niter: <class 'int'> = 5 --An edge is rewired approximately niter times.
  - connectivity: <class 'bool'> = ? --Ensure connectivity for the latticized graph when set to True.
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.
  - G: <class 'networkx.classes.graph.Graph'> = ? --An undirected graph.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.smallworld.lattice_reference(niter=<niter_value>, connectivity=<connectivity_value>, seed=<seed_value>, G=<G_variable>)

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
