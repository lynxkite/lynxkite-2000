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
