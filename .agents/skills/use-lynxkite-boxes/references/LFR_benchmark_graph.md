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
  - n: <class 'int'> = ? --Number of nodes in the created graph.
  - tau1: <class 'float'> = ? --Power law exponent for the degree distribution of the created
graph. This value must be strictly greater than one.
  - tau2: <class 'float'> = ? --Power law exponent for the community size distribution in the
created graph. This value must be strictly greater than one.
  - mu: <class 'float'> = ? --Fraction of inter-community edges incident to each node. This
value must be in the interval [0, 1].
  - average_degree: <class 'float'> = ? --Desired average degree of nodes in the created graph. This value
must be in the interval [0, *n*]. Exactly one of this and
``min_degree`` must be specified, otherwise a
:exc:`NetworkXError` is raised.
  - min_degree: <class 'int'> = ? --Minimum degree of nodes in the created graph. This value must be
in the interval [0, *n*]. Exactly one of this and
``average_degree`` must be specified, otherwise a
:exc:`NetworkXError` is raised.
  - max_degree: <class 'int'> = ? --Maximum degree of nodes in the created graph. If not specified,
this is set to ``n``, the total number of nodes in the graph.
  - min_community: <class 'int'> = ? --Minimum size of communities in the graph. If not specified, this
is set to ``min_degree``.
  - max_community: <class 'int'> = ? --Maximum size of communities in the graph. If not specified, this
is set to ``n``, the total number of nodes in the graph.
  - tol: <class 'float'> = 1e-07 --Tolerance when comparing floats, specifically when comparing
average degree values.
  - max_iters: <class 'int'> = 500 --Maximum number of iterations to try to create the community sizes,
degree distribution, and community affiliations.
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.generators.community.LFR_benchmark_graph(n=<n_value>, tau1=<tau1_value>, tau2=<tau2_value>, mu=<mu_value>, average_degree=<average_degree_value>, min_degree=<min_degree_value>, max_degree=<max_degree_value>, min_community=<min_community_value>, max_community=<max_community_value>, tol=<tol_value>, max_iters=<max_iters_value>, seed=<seed_value>)
