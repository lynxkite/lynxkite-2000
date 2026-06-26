**Minimum cycle basis:**
Returns a minimum weight cycle basis for G

Minimum weight means a cycle basis for which the total weight
(length for unweighted graphs) of all the cycles is minimum.
parameters:
  - weight: <class 'str'> = ? --name of the edge attribute to use for edge weights
  - G: <class 'networkx.classes.graph.Graph'> = ? --?

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.cycles.minimum_cycle_basis(weight=<weight_value>, G=<G_variable>)
