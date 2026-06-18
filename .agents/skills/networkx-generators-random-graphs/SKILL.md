---
name: networkx-generators-random-graphs
description: Collection of operations - Fast G(n,p) random graph, G(n,p) random graph, Dense G(n,m) random graph, G(n,m) random graph, Erdos–Renyi graph, Binomial graph, Newman–Watts–Strogatz graph, Watts–Strogatz graph, Connected Watts–Strogatz graph, Random regular graph, Barabasi–Albert graph, Dual Barabasi–Albert graph, Extended Barabasi–Albert graph, Power-law cluster graph, Random lobster graph, Random shell graph, Random power-law tree, Random power-law tree sequence
---

**Fast G(n,p) random graph:**
Returns a $G_{n,p}$ random graph, also known as an Erdős-Rényi graph or
a binomial graph.
parameters:
  - n: <class 'int'> = None - .
  - p: <class 'float'> = None - .
  - seed: int | None = None - .
  - directed: bool | None = None - .

usage:
output_variable = networkx.generators.random_graphs.fast_gnp_random_graph(n=<n_value>, p=<p_value>, seed=<seed_value>, directed=<directed_value>)

**G(n,p) random graph:**
Returns a $G_{n,p}$ random graph, also known as an Erdős-Rényi graph
or a binomial graph.

The $G_{n,p}$ model chooses each of the possible edges with probability $p$.
parameters:
  - n: <class 'int'> = None - .
  - p: <class 'float'> = None - .
  - seed: int | None = None - .
  - directed: bool | None = None - .

usage:
output_variable = networkx.generators.random_graphs.gnp_random_graph(n=<n_value>, p=<p_value>, seed=<seed_value>, directed=<directed_value>)

**Dense G(n,m) random graph:**
Returns a $G_{n,m}$ random graph.

In the $G_{n,m}$ model, a graph is chosen uniformly at random from the set
of all graphs with $n$ nodes and $m$ edges.

This algorithm should be faster than :func:`gnm_random_graph` for dense
graphs.
parameters:
  - n: <class 'int'> = None - .
  - m: <class 'int'> = None - .
  - seed: int | None = None - .

usage:
output_variable = networkx.generators.random_graphs.dense_gnm_random_graph(n=<n_value>, m=<m_value>, seed=<seed_value>)

**G(n,m) random graph:**
Returns a $G_{n,m}$ random graph.

In the $G_{n,m}$ model, a graph is chosen uniformly at random from the set
of all graphs with $n$ nodes and $m$ edges.

This algorithm should be faster than :func:`dense_gnm_random_graph` for
sparse graphs.
parameters:
  - n: <class 'int'> = None - .
  - m: <class 'int'> = None - .
  - seed: int | None = None - .
  - directed: bool | None = None - .

usage:
output_variable = networkx.generators.random_graphs.gnm_random_graph(n=<n_value>, m=<m_value>, seed=<seed_value>, directed=<directed_value>)

**Erdos–Renyi graph:**
Returns a $G_{n,p}$ random graph, also known as an Erdős-Rényi graph
or a binomial graph.

The $G_{n,p}$ model chooses each of the possible edges with probability $p$.
parameters:
  - n: <class 'int'> = None - .
  - p: <class 'float'> = None - .
  - seed: int | None = None - .
  - directed: bool | None = None - .

usage:
output_variable = networkx.generators.random_graphs.gnp_random_graph(n=<n_value>, p=<p_value>, seed=<seed_value>, directed=<directed_value>)

**Binomial graph:**
Returns a $G_{n,p}$ random graph, also known as an Erdős-Rényi graph
or a binomial graph.

The $G_{n,p}$ model chooses each of the possible edges with probability $p$.
parameters:
  - n: <class 'int'> = None - .
  - p: <class 'float'> = None - .
  - seed: int | None = None - .
  - directed: bool | None = None - .

usage:
output_variable = networkx.generators.random_graphs.gnp_random_graph(n=<n_value>, p=<p_value>, seed=<seed_value>, directed=<directed_value>)

**Newman–Watts–Strogatz graph:**
Returns a Newman–Watts–Strogatz small-world graph.
parameters:
  - n: <class 'int'> = None - .
  - k: <class 'int'> = None - .
  - p: <class 'float'> = None - .
  - seed: int | None = None - .

usage:
output_variable = networkx.generators.random_graphs.newman_watts_strogatz_graph(n=<n_value>, k=<k_value>, p=<p_value>, seed=<seed_value>)

**Watts–Strogatz graph:**
Returns a Watts–Strogatz small-world graph.
parameters:
  - n: <class 'int'> = None - .
  - k: <class 'int'> = None - .
  - p: <class 'float'> = None - .
  - seed: int | None = None - .

usage:
output_variable = networkx.generators.random_graphs.watts_strogatz_graph(n=<n_value>, k=<k_value>, p=<p_value>, seed=<seed_value>)

**Connected Watts–Strogatz graph:**
Returns a connected Watts–Strogatz small-world graph.

Attempts to generate a connected graph by repeated generation of
Watts–Strogatz small-world graphs.  An exception is raised if the maximum
number of tries is exceeded.
parameters:
  - n: <class 'int'> = None - .
  - k: <class 'int'> = None - .
  - p: <class 'float'> = None - .
  - tries: <class 'int'> = 100 - .
  - seed: int | None = None - .

usage:
output_variable = networkx.generators.random_graphs.connected_watts_strogatz_graph(n=<n_value>, k=<k_value>, p=<p_value>, tries=<tries_value>, seed=<seed_value>)

**Random regular graph:**
Returns a random $d$-regular graph on $n$ nodes.

A regular graph is a graph where each node has the same number of neighbors.

The resulting graph has no self-loops or parallel edges.
parameters:
  - d: <class 'int'> = None - .
  - n: <class 'int'> = None - .
  - seed: int | None = None - .

usage:
output_variable = networkx.generators.random_graphs.random_regular_graph(d=<d_value>, n=<n_value>, seed=<seed_value>)

**Barabasi–Albert graph:**
Returns a random graph using Barabási–Albert preferential attachment

A graph of $n$ nodes is grown by attaching new nodes each with $m$
edges that are preferentially attached to existing nodes with high degree.
parameters:
  - n: <class 'int'> = None - .
  - m: <class 'int'> = None - .
  - seed: int | None = None - .

usage:
output_variable = networkx.generators.random_graphs.barabasi_albert_graph(n=<n_value>, m=<m_value>, seed=<seed_value>)

**Dual Barabasi–Albert graph:**
Returns a random graph using dual Barabási–Albert preferential attachment

A graph of $n$ nodes is grown by attaching new nodes each with either $m_1$
edges (with probability $p$) or $m_2$ edges (with probability $1-p$) that
are preferentially attached to existing nodes with high degree.
parameters:
  - n: <class 'int'> = None - .
  - m1: <class 'int'> = None - .
  - m2: <class 'int'> = None - .
  - p: <class 'float'> = None - .
  - seed: int | None = None - .

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
  - n: <class 'int'> = None - .
  - m: <class 'int'> = None - .
  - p: <class 'float'> = None - .
  - q: <class 'float'> = None - .
  - seed: int | None = None - .

usage:
output_variable = networkx.generators.random_graphs.extended_barabasi_albert_graph(n=<n_value>, m=<m_value>, p=<p_value>, q=<q_value>, seed=<seed_value>)

**Power-law cluster graph:**
Holme and Kim algorithm for growing graphs with powerlaw
degree distribution and approximate average clustering.
parameters:
  - n: <class 'int'> = None - .
  - m: <class 'int'> = None - .
  - seed: int | None = None - .

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
  - n: <class 'int'> = None - .
  - p1: <class 'float'> = None - .
  - p2: <class 'float'> = None - .
  - seed: int | None = None - .

usage:
output_variable = networkx.generators.random_graphs.random_lobster_graph(n=<n_value>, p1=<p1_value>, p2=<p2_value>, seed=<seed_value>)

**Random shell graph:**
Returns a random shell graph for the constructor given.
parameters:
  - seed: int | None = None - .

usage:
output_variable = networkx.generators.random_graphs.random_shell_graph(seed=<seed_value>)

**Random power-law tree:**
Returns a tree with a power law degree distribution.
parameters:
  - n: <class 'int'> = None - .
  - gamma: <class 'float'> = 3 - .
  - seed: int | None = None - .
  - tries: <class 'int'> = 100 - .

usage:
output_variable = networkx.generators.random_graphs.random_powerlaw_tree(n=<n_value>, gamma=<gamma_value>, seed=<seed_value>, tries=<tries_value>)

**Random power-law tree sequence:**
Returns a degree sequence for a tree with a power law distribution.
parameters:
  - gamma: <class 'float'> = 3 - .
  - seed: int | None = None - .
  - tries: <class 'int'> = 100 - .

usage:
output_variable = networkx.generators.random_graphs.random_powerlaw_tree_sequence(gamma=<gamma_value>, seed=<seed_value>, tries=<tries_value>)
