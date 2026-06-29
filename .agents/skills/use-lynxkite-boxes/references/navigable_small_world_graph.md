**Navigable small-world graph:**
Returns a navigable small-world graph.

A navigable small-world graph is a directed grid with additional long-range
connections that are chosen randomly.

  [...] we begin with a set of nodes [...] that are identified with the set
  of lattice points in an $n \times n$ square,
  $\{(i, j): i \in \{1, 2, \ldots, n\}, j \in \{1, 2, \ldots, n\}\}$,
  and we define the *lattice distance* between two nodes $(i, j)$ and
  $(k, l)$ to be the number of "lattice steps" separating them:
  $d((i, j), (k, l)) = |k - i| + |l - j|$.

  For a universal constant $p >= 1$, the node $u$ has a directed edge to
  every other node within lattice distance $p$---these are its *local
  contacts*. For universal constants $q >= 0$ and $r >= 0$ we also
  construct directed edges from $u$ to $q$ other nodes (the *long-range
  contacts*) using independent random trials; the $i$th directed edge from
  $u$ has endpoint $v$ with probability proportional to $[d(u,v)]^{-r}$.

  -- [1]_
parameters:
  - n: <class 'int'> = ? --The length of one side of the lattice; the number of nodes in
the graph is therefore $n^2$.
  - p: <class 'int'> = 1 --The diameter of short range connections. Each node is joined with every
other node within this lattice distance.
  - q: <class 'int'> = 1 --The number of long-range connections for each node.
  - r: <class 'float'> = 2 --Exponent for decaying probability of connections.  The probability of
connecting to a node at lattice distance $d$ is $1/d^r$.
  - dim: <class 'int'> = 2 --Dimension of grid
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.geometric.navigable_small_world_graph(n=<n_value>, p=<p_value>, q=<q_value>, r=<r_value>, dim=<dim_value>, seed=<seed_value>)
