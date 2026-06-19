---
name: networkx-algorithms-distance-measures
description: Collection of operations - Eccentricity, Diameter, Harmonic diameter, Radius, Periphery, Center, Barycenter, Resistance distance, Kemeny constant, Effective graph resistance
---

**Eccentricity:**
Returns the eccentricity of nodes in G.

The eccentricity of a node v is the maximum distance from v to
all other nodes in G.
parameters:
  - weight: <class 'str'> = ? --If this is a string, then edge weights will be accessed via the
edge attribute with this key (that is, the weight of the edge
joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
such edge attribute exists, the weight of the edge is assumed to
be one.

If this is a function, the weight of an edge is the value
returned by the function. The function must accept exactly three
positional arguments: the two endpoints of an edge and the
dictionary of edge attributes for that edge. The function must
return a number.

If this is None, every edge has weight/distance/cost 1.

Weights stored as floating point values can lead to small round-off
errors in distances. Use integer weights to avoid this.

Weights should be positive, since they are distances.
  - G: <class 'networkx.classes.graph.Graph'> = ? --A graph

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.distance_measures.eccentricity(weight=<weight_value>, G=<G_variable>)

**Diameter:**
Returns the diameter of the graph G.

The diameter is the maximum eccentricity.
parameters:
  - usebounds: bool | None = ? --If `True`, use extrema bounding (see Notes) when computing the diameter
for undirected graphs. Extrema bounding may accelerate the
distance calculation for some graphs. `usebounds` is ignored if `G` is
directed or if `e` is not `None`. Default is `False`.
  - weight: <class 'str'> = ? --If this is a string, then edge weights will be accessed via the
edge attribute with this key (that is, the weight of the edge
joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
such edge attribute exists, the weight of the edge is assumed to
be one.

If this is a function, the weight of an edge is the value
returned by the function. The function must accept exactly three
positional arguments: the two endpoints of an edge and the
dictionary of edge attributes for that edge. The function must
return a number.

If this is None, every edge has weight/distance/cost 1.

Weights stored as floating point values can lead to small round-off
errors in distances. Use integer weights to avoid this.

Weights should be positive, since they are distances.
  - G: <class 'networkx.classes.graph.Graph'> = ? --A graph

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.distance_measures.diameter(usebounds=<usebounds_value>, weight=<weight_value>, G=<G_variable>)

**Harmonic diameter:**
Returns the harmonic diameter of the graph G.

The harmonic diameter of a graph is the harmonic mean of the distances
between all pairs of distinct vertices. Graphs that are not strongly
connected have infinite diameter and mean distance, making such
measures not useful. Restricting the diameter or mean distance to
finite distances yields paradoxical values (e.g., a perfect match
would have diameter one). The harmonic mean handles gracefully
infinite distances (e.g., a perfect match has harmonic diameter equal
to the number of vertices minus one), making it possible to assign a
meaningful value to all graphs.

Note that in [1] the harmonic diameter is called "connectivity length":
however, "harmonic diameter" is a more standard name from the
theory of metric spaces. The name "harmonic mean distance" is perhaps
a more descriptive name, but is not used in the literature, so we use the
name "harmonic diameter" here.
parameters:
  - weight: <class 'str'> = ? --If None, every edge has weight/distance 1.
If a string, use this edge attribute as the edge weight.
Any edge attribute not present defaults to 1.
If a function, the weight of an edge is the value returned by the function.
The function must accept exactly three positional arguments:
the two endpoints of an edge and the dictionary of edge attributes for
that edge. The function must return a number.
  - G: <class 'networkx.classes.graph.Graph'> = ? --A graph

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.distance_measures.harmonic_diameter(weight=<weight_value>, G=<G_variable>)

**Radius:**
Returns the radius of the graph G.

The radius is the minimum eccentricity.
parameters:
  - usebounds: bool | None = ? --If `True`, use extrema bounding (see Notes) when computing the radius
for undirected graphs. Extrema bounding may accelerate the
distance calculation for some graphs. `usebounds` is ignored if `G` is
directed or if `e` is not `None`. Default is `False`.
  - weight: <class 'str'> = ? --If this is a string, then edge weights will be accessed via the
edge attribute with this key (that is, the weight of the edge
joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
such edge attribute exists, the weight of the edge is assumed to
be one.

If this is a function, the weight of an edge is the value
returned by the function. The function must accept exactly three
positional arguments: the two endpoints of an edge and the
dictionary of edge attributes for that edge. The function must
return a number.

If this is None, every edge has weight/distance/cost 1.

Weights stored as floating point values can lead to small round-off
errors in distances. Use integer weights to avoid this.

Weights should be positive, since they are distances.
  - G: <class 'networkx.classes.graph.Graph'> = ? --A graph

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.distance_measures.radius(usebounds=<usebounds_value>, weight=<weight_value>, G=<G_variable>)

**Periphery:**
Returns the periphery of the graph G.

The periphery is the set of nodes with eccentricity equal to the diameter.
parameters:
  - usebounds: bool | None = ? --If `True`, use extrema bounding (see Notes) when computing the periphery
for undirected graphs. Extrema bounding may accelerate the
distance calculation for some graphs. `usebounds` is ignored if `G` is
directed or if `e` is not `None`. Default is `False`.
  - weight: <class 'str'> = ? --If this is a string, then edge weights will be accessed via the
edge attribute with this key (that is, the weight of the edge
joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
such edge attribute exists, the weight of the edge is assumed to
be one.

If this is a function, the weight of an edge is the value
returned by the function. The function must accept exactly three
positional arguments: the two endpoints of an edge and the
dictionary of edge attributes for that edge. The function must
return a number.

If this is None, every edge has weight/distance/cost 1.

Weights stored as floating point values can lead to small round-off
errors in distances. Use integer weights to avoid this.

Weights should be positive, since they are distances.
  - G: <class 'networkx.classes.graph.Graph'> = ? --A graph

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.distance_measures.periphery(usebounds=<usebounds_value>, weight=<weight_value>, G=<G_variable>)

**Center:**
Returns the center of the graph G.

The center is the set of nodes with eccentricity equal to radius.
parameters:
  - usebounds: bool | None = ? --If `True`, use extrema bounding (see Notes) when computing the center
for undirected graphs. Extrema bounding may accelerate the
distance calculation for some graphs. `usebounds` is ignored if `G` is
directed or if `e` is not `None`. Default is `False`.
  - weight: <class 'str'> = ? --If this is a string, then edge weights will be accessed via the
edge attribute with this key (that is, the weight of the edge
joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
such edge attribute exists, the weight of the edge is assumed to
be one.

If this is a function, the weight of an edge is the value
returned by the function. The function must accept exactly three
positional arguments: the two endpoints of an edge and the
dictionary of edge attributes for that edge. The function must
return a number.

If this is None, every edge has weight/distance/cost 1.

Weights stored as floating point values can lead to small round-off
errors in distances. Use integer weights to avoid this.

Weights should be positive, since they are distances.
  - G: <class 'networkx.classes.graph.Graph'> = ? --A graph

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.distance_measures.center(usebounds=<usebounds_value>, weight=<weight_value>, G=<G_variable>)

**Barycenter:**
Calculate barycenter of a connected graph, optionally with edge weights.

The :dfn:`barycenter` a
:func:`connected <networkx.algorithms.components.is_connected>` graph
:math:`G` is the subgraph induced by the set of its nodes :math:`v`
minimizing the objective function

.. math::

    \sum_{u \in V(G)} d_G(u, v),

where :math:`d_G` is the (possibly weighted) :func:`path length
<networkx.algorithms.shortest_paths.generic.shortest_path_length>`.
The barycenter is also called the :dfn:`median`. See [West01]_, p. 78.
parameters:
  - weight: str | None = ? --Passed through to
:func:`~networkx.algorithms.shortest_paths.generic.shortest_path_length`.
  - G: <class 'networkx.classes.graph.Graph'> = ? --The connected graph :math:`G`.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.distance_measures.barycenter(weight=<weight_value>, G=<G_variable>)

**Resistance distance:**
Returns the resistance distance between pairs of nodes in graph G.

The resistance distance between two nodes of a graph is akin to treating
the graph as a grid of resistors with a resistance equal to the provided
weight [1]_, [2]_.

If weight is not provided, then a weight of 1 is used for all edges.

If two nodes are the same, the resistance distance is zero.
parameters:
  - weight: str | None = ? --The edge data key used to compute the resistance distance.
If None, then each edge has weight 1.
  - invert_weight: <class 'bool'> = ? --Proper calculation of resistance distance requires building the
Laplacian matrix with the reciprocal of the weight. Not required
if the weight is already inverted. Weight cannot be zero.
  - G: <class 'networkx.classes.graph.Graph'> = ? --A graph

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.distance_measures.resistance_distance(weight=<weight_value>, invert_weight=<invert_weight_value>, G=<G_variable>)

**Kemeny constant:**
Returns the Kemeny constant of the given graph.

The *Kemeny constant* (or Kemeny's constant) of a graph `G`
can be computed by regarding the graph as a Markov chain.
The Kemeny constant is then the expected number of time steps
to transition from a starting state i to a random destination state
sampled from the Markov chain's stationary distribution.
The Kemeny constant is independent of the chosen initial state [1]_.

The Kemeny constant measures the time needed for spreading
across a graph. Low values indicate a closely connected graph
whereas high values indicate a spread-out graph.

If weight is not provided, then a weight of 1 is used for all edges.

Since `G` represents a Markov chain, the weights must be positive.
parameters:
  - weight: str | None = ? --The edge data key used to compute the Kemeny constant.
If None, then each edge has weight 1.
  - G: <class 'networkx.classes.graph.Graph'> = ? --?

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.distance_measures.kemeny_constant(weight=<weight_value>, G=<G_variable>)

**Effective graph resistance:**
Returns the Effective graph resistance of G.

Also known as the Kirchhoff index.

The effective graph resistance is defined as the sum
of the resistance distance of every node pair in G [1]_.

If weight is not provided, then a weight of 1 is used for all edges.

The effective graph resistance of a disconnected graph is infinite.
parameters:
  - weight: str | None = ? --The edge data key used to compute the effective graph resistance.
If None, then each edge has weight 1.
  - invert_weight: <class 'bool'> = ? --Proper calculation of resistance distance requires building the
Laplacian matrix with the reciprocal of the weight. Not required
if the weight is already inverted. Weight cannot be zero.
  - G: <class 'networkx.classes.graph.Graph'> = ? --A graph

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.distance_measures.effective_graph_resistance(weight=<weight_value>, invert_weight=<invert_weight_value>, G=<G_variable>)
