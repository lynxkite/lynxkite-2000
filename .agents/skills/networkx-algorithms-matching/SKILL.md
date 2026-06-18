---
name: networkx-algorithms-matching
description: Collection of operations - Is matching, Is maximal matching, Is perfect matching, Max weight matching, Min weight matching, Maximal matching
---

**Is matching:**
Return True if ``matching`` is a valid matching of ``G``

A *matching* in a graph is a set of edges in which no two distinct
edges share a common endpoint. Each node is incident to at most one
edge in the matching. The edges are said to be independent.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.algorithms.matching.is_matching(G=<G_variable>)

**Is maximal matching:**
Return True if ``matching`` is a maximal matching of ``G``

A *maximal matching* in a graph is a matching in which adding any
edge would cause the set to no longer be a valid matching.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.algorithms.matching.is_maximal_matching(G=<G_variable>)

**Is perfect matching:**
Return True if ``matching`` is a perfect matching for ``G``

A *perfect matching* in a graph is a matching in which exactly one edge
is incident upon each vertex.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.algorithms.matching.is_perfect_matching(G=<G_variable>)

**Max weight matching:**
Compute a maximum-weighted matching of G.

A matching is a subset of edges in which no node occurs more than once.
The weight of a matching is the sum of the weights of its edges.
A maximal matching cannot add more edges and still be a matching.
The cardinality of a matching is the number of matched edges.
parameters:
  - maxcardinality: bool | None = None - .
  - weight: str | None = weight - .
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.algorithms.matching.max_weight_matching(maxcardinality=<maxcardinality_value>, weight=<weight_value>, G=<G_variable>)

**Min weight matching:**
Compute a minimum-weight maximum-cardinality matching of `G`.

The minimum-weight maximum-cardinality matching is the matching
that has the minimum weight among all maximum-cardinality matchings.

Use the maximum-weight algorithm with edge weights subtracted
from the maximum weight of all edges.

A matching is a subset of edges in which no node occurs more than once.
The weight of a matching is the sum of the weights of its edges.
A maximal matching cannot add more edges and still be a matching.
The cardinality of a matching is the number of matched edges.

This method replaces the edge weights with 1 plus the maximum edge weight
minus the original edge weight.

new_weight = (max_weight + 1) - edge_weight

then runs :func:`max_weight_matching` with the new weights.
The max weight matching with these new weights corresponds
to the min weight matching using the original weights.
Adding 1 to the max edge weight keeps all edge weights positive
and as integers if they started as integers.

Read the documentation of `max_weight_matching` for more information.
parameters:
  - weight: str | None = weight - .
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.algorithms.matching.min_weight_matching(weight=<weight_value>, G=<G_variable>)

**Maximal matching:**
Find a maximal matching in the graph.

A matching is a subset of edges in which no node occurs more than once.
A maximal matching cannot add more edges and still be a matching.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.algorithms.matching.maximal_matching(G=<G_variable>)
