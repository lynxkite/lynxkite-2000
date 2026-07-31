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
  - weight: str | None = weight --Edge data key corresponding to the edge weight.
If key not found, uses 1 as weight.
  - G: <class 'networkx.classes.graph.Graph'> = ? --Undirected graph
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.matching.min_weight_matching(weight=<weight_value>, G=<G_variable>)
