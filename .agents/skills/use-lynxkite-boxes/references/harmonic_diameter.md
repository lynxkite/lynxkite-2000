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
