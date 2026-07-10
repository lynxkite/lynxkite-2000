**Flow hierarchy:**
Returns the flow hierarchy of a directed network.

Flow hierarchy is defined as the fraction of edges not participating
in cycles in a directed graph [1]_.
parameters:
  - weight: str | None = ? --Attribute to use for edge weights. If None the weight defaults to 1.
  - G: <class 'networkx.classes.graph.Graph'> = ? --A directed graph
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.hierarchy.flow_hierarchy(weight=<weight_value>, G=<G_variable>)
