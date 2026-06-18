---
name: networkx-generators-community
description: Collection of operations - Caveman graph, Connected caveman graph, Relaxed caveman graph, Random partition graph, Planted partition graph, Gaussian random partition graph, Ring of cliques, Windmill graph, Stochastic block model, LFR benchmark graph
---

**Caveman graph:**
Returns a caveman graph of `l` cliques of size `k`.
parameters:
  - l: <class 'int'> = None - .
  - k: <class 'int'> = None - .

usage:
output_variable = networkx.generators.community.caveman_graph(l=<l_value>, k=<k_value>)

**Connected caveman graph:**
Returns a connected caveman graph of `l` cliques of size `k`.

The connected caveman graph is formed by creating `n` cliques of size
`k`, then a single edge in each clique is rewired to a node in an
adjacent clique.
parameters:
  - l: <class 'int'> = None - .
  - k: <class 'int'> = None - .

usage:
output_variable = networkx.generators.community.connected_caveman_graph(l=<l_value>, k=<k_value>)

**Relaxed caveman graph:**
Returns a relaxed caveman graph.

A relaxed caveman graph starts with `l` cliques of size `k`.  Edges are
then randomly rewired with probability `p` to link different cliques.
parameters:
  - l: <class 'int'> = None - .
  - k: <class 'int'> = None - .
  - p: <class 'float'> = None - .
  - seed: int | None = None - .

usage:
output_variable = networkx.generators.community.relaxed_caveman_graph(l=<l_value>, k=<k_value>, p=<p_value>, seed=<seed_value>)

**Random partition graph:**
Returns the random partition graph with a partition of sizes.

A partition graph is a graph of communities with sizes defined by
s in sizes. Nodes in the same group are connected with probability
p_in and nodes of different groups are connected with probability
p_out.
parameters:
  - p_in: <class 'float'> = None - .
  - p_out: <class 'float'> = None - .
  - seed: int | None = None - .

usage:
output_variable = networkx.generators.community.random_partition_graph(p_in=<p_in_value>, p_out=<p_out_value>, seed=<seed_value>)

**Planted partition graph:**
Returns the planted l-partition graph.

This model partitions a graph with n=l*k vertices in
l groups with k vertices each. Vertices of the same
group are linked with a probability p_in, and vertices
of different groups are linked with probability p_out.
parameters:
  - l: <class 'int'> = None - .
  - k: <class 'int'> = None - .
  - p_in: <class 'float'> = None - .
  - p_out: <class 'float'> = None - .
  - seed: int | None = None - .

usage:
output_variable = networkx.generators.community.planted_partition_graph(l=<l_value>, k=<k_value>, p_in=<p_in_value>, p_out=<p_out_value>, seed=<seed_value>)

**Gaussian random partition graph:**
Generate a Gaussian random partition graph.

A Gaussian random partition graph is created by creating k partitions
each with a size drawn from a normal distribution with mean s and variance
s/v. Nodes are connected within clusters with probability p_in and
between clusters with probability p_out[1]
parameters:
  - n: <class 'int'> = None - .
  - s: <class 'float'> = None - .
  - v: <class 'float'> = None - .
  - p_in: <class 'float'> = None - .
  - p_out: <class 'float'> = None - .
  - seed: int | None = None - .

usage:
output_variable = networkx.generators.community.gaussian_random_partition_graph(n=<n_value>, s=<s_value>, v=<v_value>, p_in=<p_in_value>, p_out=<p_out_value>, seed=<seed_value>)

**Ring of cliques:**
Defines a "ring of cliques" graph.

A ring of cliques graph is consisting of cliques, connected through single
links. Each clique is a complete graph.
parameters:
  - num_cliques: <class 'int'> = None - .
  - clique_size: <class 'int'> = None - .

usage:
output_variable = networkx.generators.community.ring_of_cliques(num_cliques=<num_cliques_value>, clique_size=<clique_size_value>)

**Windmill graph:**
Generate a windmill graph.
A windmill graph is a graph of `n` cliques each of size `k` that are all
joined at one node.
It can be thought of as taking a disjoint union of `n` cliques of size `k`,
selecting one point from each, and contracting all of the selected points.
Alternatively, one could generate `n` cliques of size `k-1` and one node
that is connected to all other nodes in the graph.
parameters:
  - n: <class 'int'> = None - .
  - k: <class 'int'> = None - .

usage:
output_variable = networkx.generators.community.windmill_graph(n=<n_value>, k=<k_value>)

**Stochastic block model:**
Returns a stochastic block model graph.

This model partitions the nodes in blocks of arbitrary sizes, and places
edges between pairs of nodes independently, with a probability that depends
on the blocks.
parameters:
  - seed: int | None = None - .

usage:
output_variable = networkx.generators.community.stochastic_block_model(seed=<seed_value>)

**LFR benchmark graph:**
Returns the LFR benchmark graph.

This algorithm proceeds as follows:

1) Find a degree sequence with a power law distribution, and minimum
   value ``min_degree``, which has approximate average degree
   ``average_degree``. This is accomplished by either

   a) specifying ``min_degree`` and not ``average_degree``,
   b) specifying ``average_degree`` and not ``min_degree``, in which
      case a suitable minimum degree will be found.

   ``max_degree`` can also be specified, otherwise it will be set to
   ``n``. Each node *u* will have $\mu \mathrm{deg}(u)$ edges
   joining it to nodes in communities other than its own and $(1 -
   \mu) \mathrm{deg}(u)$ edges joining it to nodes in its own
   community.
2) Generate community sizes according to a power law distribution
   with exponent ``tau2``. If ``min_community`` and
   ``max_community`` are not specified they will be selected to be
   ``min_degree`` and ``max_degree``, respectively.  Community sizes
   are generated until the sum of their sizes equals ``n``.
3) Each node will be randomly assigned a community with the
   condition that the community is large enough for the node's
   intra-community degree, $(1 - \mu) \mathrm{deg}(u)$ as
   described in step 2. If a community grows too large, a random node
   will be selected for reassignment to a new community, until all
   nodes have been assigned a community.
4) Each node *u* then adds $(1 - \mu) \mathrm{deg}(u)$
   intra-community edges and $\mu \mathrm{deg}(u)$ inter-community
   edges.
parameters:
  - n: <class 'int'> = None - .
  - tau1: <class 'float'> = None - .
  - tau2: <class 'float'> = None - .
  - mu: <class 'float'> = None - .
  - average_degree: <class 'float'> = None - .
  - min_degree: <class 'int'> = None - .
  - max_degree: <class 'int'> = None - .
  - min_community: <class 'int'> = None - .
  - max_community: <class 'int'> = None - .
  - tol: <class 'float'> = 1e-07 - .
  - max_iters: <class 'int'> = 500 - .
  - seed: int | None = None - .

usage:
output_variable = networkx.generators.community.LFR_benchmark_graph(n=<n_value>, tau1=<tau1_value>, tau2=<tau2_value>, mu=<mu_value>, average_degree=<average_degree_value>, min_degree=<min_degree_value>, max_degree=<max_degree_value>, min_community=<min_community_value>, max_community=<max_community_value>, tol=<tol_value>, max_iters=<max_iters_value>, seed=<seed_value>)
