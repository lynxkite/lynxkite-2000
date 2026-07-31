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
