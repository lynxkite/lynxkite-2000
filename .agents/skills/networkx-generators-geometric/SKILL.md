---
name: networkx-generators-geometric
description: Collection of operations - Geometric edges, Navigable small-world graph, Random geometric graph, Soft random geometric graph, Thresholded random geometric graph
---

**Geometric edges:**
Returns edge list of node pairs within `radius` of each other.
parameters:
  - radius: <class 'float'> = ? --The distance threshold. Edges are included in the edge list if the
distance between the two nodes is less than `radius`.
  - G: <class 'networkx.classes.graph.Graph'> = ? --The graph from which to generate the edge list. The nodes in `G` should
have an attribute ``pos`` corresponding to the node position, which is
used to compute the distance to other nodes.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.geometric.geometric_edges(radius=<radius_value>, G=<G_variable>)

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

**Random geometric graph:**
Returns a random geometric graph in the unit cube of dimensions `dim`.

The random geometric graph model places `n` nodes uniformly at
random in the unit cube. Two nodes are joined by an edge if the
distance between the nodes is at most `radius`.

Edges are determined using a KDTree when SciPy is available.
This reduces the time complexity from $O(n^2)$ to $O(n)$.
parameters:
  - radius: <class 'float'> = ? --Distance threshold value
  - dim: int | None = 2 --Dimension of graph
  - p: float | None = 2 --Which Minkowski distance metric to use.  `p` has to meet the condition
``1 <= p <= infinity``.

If this argument is not specified, the :math:`L^2` metric
(the Euclidean distance metric), p = 2 is used.
This should not be confused with the `p` of an Erdős-Rényi random
graph, which represents probability.
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.geometric.random_geometric_graph(radius=<radius_value>, dim=<dim_value>, p=<p_value>, seed=<seed_value>)

**Soft random geometric graph:**
Returns a soft random geometric graph in the unit cube.

The soft random geometric graph [1] model places `n` nodes uniformly at
random in the unit cube in dimension `dim`. Two nodes of distance, `dist`,
computed by the `p`-Minkowski distance metric are joined by an edge with
probability `p_dist` if the computed distance metric value of the nodes
is at most `radius`, otherwise they are not joined.

Edges within `radius` of each other are determined using a KDTree when
SciPy is available. This reduces the time complexity from :math:`O(n^2)`
to :math:`O(n)`.
parameters:
  - radius: <class 'float'> = ? --Distance threshold value
  - dim: int | None = 2 --Dimension of graph
  - p: float | None = 2 --Which Minkowski distance metric to use.
`p` has to meet the condition ``1 <= p <= infinity``.

If this argument is not specified, the :math:`L^2` metric
(the Euclidean distance metric), p = 2 is used.

This should not be confused with the `p` of an Erdős-Rényi random
graph, which represents probability.
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.geometric.soft_random_geometric_graph(radius=<radius_value>, dim=<dim_value>, p=<p_value>, seed=<seed_value>)

**Thresholded random geometric graph:**
Returns a thresholded random geometric graph in the unit cube.

The thresholded random geometric graph [1] model places `n` nodes
uniformly at random in the unit cube of dimensions `dim`. Each node
`u` is assigned a weight :math:`w_u`. Two nodes `u` and `v` are
joined by an edge if they are within the maximum connection distance,
`radius` computed by the `p`-Minkowski distance and the summation of
weights :math:`w_u` + :math:`w_v` is greater than or equal
to the threshold parameter `theta`.

Edges within `radius` of each other are determined using a KDTree when
SciPy is available. This reduces the time complexity from :math:`O(n^2)`
to :math:`O(n)`.
parameters:
  - radius: <class 'float'> = ? --Distance threshold value
  - theta: <class 'float'> = ? --Threshold value
  - dim: int | None = 2 --Dimension of graph
  - p: float | None = 2 --Which Minkowski distance metric to use.  `p` has to meet the condition
``1 <= p <= infinity``.

If this argument is not specified, the :math:`L^2` metric
(the Euclidean distance metric), p = 2 is used.

This should not be confused with the `p` of an Erdős-Rényi random
graph, which represents probability.
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.geometric.thresholded_random_geometric_graph(radius=<radius_value>, theta=<theta_value>, dim=<dim_value>, p=<p_value>, seed=<seed_value>)
