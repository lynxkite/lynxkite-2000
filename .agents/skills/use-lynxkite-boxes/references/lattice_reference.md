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
