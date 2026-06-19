---
name: networkx-algorithms-distance-regular
description: Collection of operations - Is distance regular, Is strongly regular, Intersection array
---

**Is distance regular:**
Returns True if the graph is distance regular, False otherwise.

A connected graph G is distance-regular if for any nodes x,y
and any integers i,j=0,1,...,d (where d is the graph
diameter), the number of vertices at distance i from x and
distance j from y depends only on i,j and the graph distance
between x and y, independently of the choice of x and y.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --?

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.distance_regular.is_distance_regular(G=<G_variable>)

**Is strongly regular:**
Returns True if and only if the given graph is strongly
regular.

An undirected graph is *strongly regular* if

* it is regular,
* each pair of adjacent vertices has the same number of neighbors in
  common,
* each pair of nonadjacent vertices has the same number of neighbors
  in common.

Each strongly regular graph is a distance-regular graph.
Conversely, if a distance-regular graph has diameter two, then it is
a strongly regular graph. For more information on distance-regular
graphs, see :func:`is_distance_regular`.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --An undirected graph.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.distance_regular.is_strongly_regular(G=<G_variable>)

**Intersection array:**
Returns the intersection array of a distance-regular graph.

Given a distance-regular graph G with integers b_i, c_i,i = 0,....,d
such that for any 2 vertices x,y in G at a distance i=d(x,y), there
are exactly c_i neighbors of y at a distance of i-1 from x and b_i
neighbors of y at a distance of i+1 from x.

A distance regular graph's intersection array is given by,
[b_0,b_1,.....b_{d-1};c_1,c_2,.....c_d]
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --?

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.distance_regular.intersection_array(G=<G_variable>)
