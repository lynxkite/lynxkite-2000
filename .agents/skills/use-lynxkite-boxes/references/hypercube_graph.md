**Hypercube graph:**
Returns the *n*-dimensional hypercube graph.

The *n*-dimensional hypercube graph [1]_ has ``2**n`` nodes, each represented as
a binary integer in the form of a tuple of 0's and 1's. Edges exist between
nodes that differ in exactly one bit.
parameters:
  - n: <class 'int'> = ? --Dimension of the hypercube, must be a positive integer.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.lattice.hypercube_graph(n=<n_value>)
