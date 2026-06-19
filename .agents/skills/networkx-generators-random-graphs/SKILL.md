---
name: networkx-generators-random-graphs
description: Collection of operations - Fast G(n,p) random graph, G(n,p) random graph, Dense G(n,m) random graph, G(n,m) random graph, Erdos–Renyi graph, Binomial graph, Newman–Watts–Strogatz graph, Watts–Strogatz graph, Connected Watts–Strogatz graph, Random regular graph, Barabasi–Albert graph, Dual Barabasi–Albert graph, Extended Barabasi–Albert graph, Power-law cluster graph, Random lobster graph, Random shell graph, Random power-law tree, Random power-law tree sequence
---

**Fast G(n,p) random graph:**
Returns a $G_{n,p}$ random graph, also known as an Erdős-Rényi graph or
a binomial graph.
parameters:
  - n: <class 'int'> = ? --The number of nodes.
  - p: <class 'float'> = ? --Probability for edge creation.
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.
  - directed: bool | None = ? --If True, this function returns a directed graph.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.random_graphs.fast_gnp_random_graph(n=<n_value>, p=<p_value>, seed=<seed_value>, directed=<directed_value>)

**G(n,p) random graph:**
Returns a $G_{n,p}$ random graph, also known as an Erdős-Rényi graph
or a binomial graph.

The $G_{n,p}$ model chooses each of the possible edges with probability $p$.
parameters:
  - n: <class 'int'> = ? --The number of nodes.
  - p: <class 'float'> = ? --Probability for edge creation.
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.
  - directed: bool | None = ? --If True, this function returns a directed graph.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.random_graphs.gnp_random_graph(n=<n_value>, p=<p_value>, seed=<seed_value>, directed=<directed_value>)

**Dense G(n,m) random graph:**
Returns a $G_{n,m}$ random graph.

In the $G_{n,m}$ model, a graph is chosen uniformly at random from the set
of all graphs with $n$ nodes and $m$ edges.

This algorithm should be faster than :func:`gnm_random_graph` for dense
graphs.
parameters:
  - n: <class 'int'> = ? --The number of nodes.
  - m: <class 'int'> = ? --The number of edges.
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.random_graphs.dense_gnm_random_graph(n=<n_value>, m=<m_value>, seed=<seed_value>)

**G(n,m) random graph:**
Returns a $G_{n,m}$ random graph.

In the $G_{n,m}$ model, a graph is chosen uniformly at random from the set
of all graphs with $n$ nodes and $m$ edges.

This algorithm should be faster than :func:`dense_gnm_random_graph` for
sparse graphs.
parameters:
  - n: <class 'int'> = ? --The number of nodes.
  - m: <class 'int'> = ? --The number of edges.
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.
  - directed: bool | None = ? --If True return a directed graph

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.random_graphs.gnm_random_graph(n=<n_value>, m=<m_value>, seed=<seed_value>, directed=<directed_value>)

**Erdos–Renyi graph:**
Returns a $G_{n,p}$ random graph, also known as an Erdős-Rényi graph
or a binomial graph.

The $G_{n,p}$ model chooses each of the possible edges with probability $p$.
parameters:
  - n: <class 'int'> = ? --The number of nodes.
  - p: <class 'float'> = ? --Probability for edge creation.
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.
  - directed: bool | None = ? --If True, this function returns a directed graph.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.random_graphs.gnp_random_graph(n=<n_value>, p=<p_value>, seed=<seed_value>, directed=<directed_value>)

**Binomial graph:**
Returns a $G_{n,p}$ random graph, also known as an Erdős-Rényi graph
or a binomial graph.

The $G_{n,p}$ model chooses each of the possible edges with probability $p$.
parameters:
  - n: <class 'int'> = ? --The number of nodes.
  - p: <class 'float'> = ? --Probability for edge creation.
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.
  - directed: bool | None = ? --If True, this function returns a directed graph.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.random_graphs.gnp_random_graph(n=<n_value>, p=<p_value>, seed=<seed_value>, directed=<directed_value>)

**Newman–Watts–Strogatz graph:**
Returns a Newman–Watts–Strogatz small-world graph.
parameters:
  - n: <class 'int'> = ? --The number of nodes.
  - k: <class 'int'> = ? --Each node is joined with its `k` nearest neighbors in a ring
topology.
  - p: <class 'float'> = ? --The probability of adding a new edge for each edge.
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.random_graphs.newman_watts_strogatz_graph(n=<n_value>, k=<k_value>, p=<p_value>, seed=<seed_value>)

**Watts–Strogatz graph:**
Returns a Watts–Strogatz small-world graph.
parameters:
  - n: <class 'int'> = ? --The number of nodes
  - k: <class 'int'> = ? --Each node is joined with its `k` nearest neighbors in a ring
topology.
  - p: <class 'float'> = ? --The probability of rewiring each edge
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.random_graphs.watts_strogatz_graph(n=<n_value>, k=<k_value>, p=<p_value>, seed=<seed_value>)

**Connected Watts–Strogatz graph:**
Returns a connected Watts–Strogatz small-world graph.

Attempts to generate a connected graph by repeated generation of
Watts–Strogatz small-world graphs.  An exception is raised if the maximum
number of tries is exceeded.
parameters:
  - n: <class 'int'> = ? --The number of nodes
  - k: <class 'int'> = ? --Each node is joined with its `k` nearest neighbors in a ring
topology.
  - p: <class 'float'> = ? --The probability of rewiring each edge
  - tries: <class 'int'> = 100 --Number of attempts to generate a connected graph.
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.random_graphs.connected_watts_strogatz_graph(n=<n_value>, k=<k_value>, p=<p_value>, tries=<tries_value>, seed=<seed_value>)

**Random regular graph:**
Returns a random $d$-regular graph on $n$ nodes.

A regular graph is a graph where each node has the same number of neighbors.

The resulting graph has no self-loops or parallel edges.
parameters:
  - d: <class 'int'> = ? --The degree of each node.
  - n: <class 'int'> = ? --The number of nodes. The value of $n \times d$ must be even.
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.random_graphs.random_regular_graph(d=<d_value>, n=<n_value>, seed=<seed_value>)

**Barabasi–Albert graph:**
Returns a random graph using Barabási–Albert preferential attachment

A graph of $n$ nodes is grown by attaching new nodes each with $m$
edges that are preferentially attached to existing nodes with high degree.
parameters:
  - n: <class 'int'> = ? --Number of nodes
  - m: <class 'int'> = ? --Number of edges to attach from a new node to existing nodes
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.random_graphs.barabasi_albert_graph(n=<n_value>, m=<m_value>, seed=<seed_value>)

**Dual Barabasi–Albert graph:**
Returns a random graph using dual Barabási–Albert preferential attachment

A graph of $n$ nodes is grown by attaching new nodes each with either $m_1$
edges (with probability $p$) or $m_2$ edges (with probability $1-p$) that
are preferentially attached to existing nodes with high degree.
parameters:
  - n: <class 'int'> = ? --Number of nodes
  - m1: <class 'int'> = ? --Number of edges to link each new node to existing nodes with probability $p$
  - m2: <class 'int'> = ? --Number of edges to link each new node to existing nodes with probability $1-p$
  - p: <class 'float'> = ? --The probability of attaching $m_1$ edges (as opposed to $m_2$ edges)
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.random_graphs.dual_barabasi_albert_graph(n=<n_value>, m1=<m1_value>, m2=<m2_value>, p=<p_value>, seed=<seed_value>)

**Extended Barabasi–Albert graph:**
Returns an extended Barabási–Albert model graph.

An extended Barabási–Albert model graph is a random graph constructed
using preferential attachment. The extended model allows new edges,
rewired edges or new nodes. Based on the probabilities $p$ and $q$
with $p + q < 1$, the growing behavior of the graph is determined as:

1) With $p$ probability, $m$ new edges are added to the graph,
starting from randomly chosen existing nodes and attached preferentially at the
other end.

2) With $q$ probability, $m$ existing edges are rewired
by randomly choosing an edge and rewiring one end to a preferentially chosen node.

3) With $(1 - p - q)$ probability, $m$ new nodes are added to the graph
with edges attached preferentially.

When $p = q = 0$, the model behaves just like the Barabási–Alber model.
parameters:
  - n: <class 'int'> = ? --Number of nodes
  - m: <class 'int'> = ? --Number of edges with which a new node attaches to existing nodes
  - p: <class 'float'> = ? --Probability value for adding an edge between existing nodes. p + q < 1
  - q: <class 'float'> = ? --Probability value of rewiring of existing edges. p + q < 1
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.random_graphs.extended_barabasi_albert_graph(n=<n_value>, m=<m_value>, p=<p_value>, q=<q_value>, seed=<seed_value>)

**Power-law cluster graph:**
Holme and Kim algorithm for growing graphs with powerlaw
degree distribution and approximate average clustering.
parameters:
  - n: <class 'int'> = ? --the number of nodes
  - m: <class 'int'> = ? --the number of random edges to add for each new node
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.random_graphs.powerlaw_cluster_graph(n=<n_value>, m=<m_value>, seed=<seed_value>)

**Random lobster graph:**
Returns a random lobster graph.

A lobster is a tree that reduces to a caterpillar when pruning all
leaf nodes. A caterpillar is a tree that reduces to a path graph
when pruning all leaf nodes; setting `p2` to zero produces a caterpillar.

This implementation iterates on the probabilities `p1` and `p2` to add
edges at levels 1 and 2, respectively. Graphs are therefore constructed
iteratively with uniform randomness at each level rather than being selected
uniformly at random from the set of all possible lobsters.
parameters:
  - n: <class 'int'> = ? --The expected number of nodes in the backbone
  - p1: <class 'float'> = ? --Probability of adding an edge to the backbone
  - p2: <class 'float'> = ? --Probability of adding an edge one level beyond backbone
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.random_graphs.random_lobster_graph(n=<n_value>, p1=<p1_value>, p2=<p2_value>, seed=<seed_value>)

**Random shell graph:**
Returns a random shell graph for the constructor given.
parameters:
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.random_graphs.random_shell_graph(seed=<seed_value>)

**Random power-law tree:**
Returns a tree with a power law degree distribution.
parameters:
  - n: <class 'int'> = ? --The number of nodes.
  - gamma: <class 'float'> = 3 --Exponent of the power law.
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.
  - tries: <class 'int'> = 100 --Number of attempts to adjust the sequence to make it a tree.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.random_graphs.random_powerlaw_tree(n=<n_value>, gamma=<gamma_value>, seed=<seed_value>, tries=<tries_value>)

**Random power-law tree sequence:**
Returns a degree sequence for a tree with a power law distribution.
parameters:
  - gamma: <class 'float'> = 3 --Exponent of the power law.
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.
  - tries: <class 'int'> = 100 --Number of attempts to adjust the sequence to make it a tree.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.random_graphs.random_powerlaw_tree_sequence(gamma=<gamma_value>, seed=<seed_value>, tries=<tries_value>)
