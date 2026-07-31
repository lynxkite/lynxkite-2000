**Antichains:**
Generates antichains from a directed acyclic graph (DAG).

An antichain is a subset of a partially ordered set such that any
two elements in the subset are incomparable.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --A directed acyclic graph (DAG)
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.dag.antichains(G=<G_variable>)
