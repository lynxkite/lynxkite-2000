**Topological generations:**
Stratifies a DAG into generations.

A topological generation is node collection in which ancestors of a node in each
generation are guaranteed to be in a previous generation, and any descendants of
a node are guaranteed to be in a following generation. Nodes are guaranteed to
be in the earliest possible generation that they can belong to.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --A directed acyclic graph (DAG)
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.dag.topological_generations(G=<G_variable>)
