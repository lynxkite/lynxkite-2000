**Heawood graph:**
Returns the Heawood Graph, a (3,6) cage.

The Heawood Graph is an undirected graph with 14 nodes and 21 edges,
named after Percy John Heawood [1]_.
It is cubic symmetric, nonplanar, Hamiltonian, and can be represented
in LCF notation as ``[5,-5]^7`` [2]_.
It is the unique (3,6)-cage: the regular cubic graph of girth 6 with
minimal number of vertices [3]_.
parameters:
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.generators.small.heawood_graph()
