**Minimum spanning arborescence:**
Returns a minimum spanning arborescence from G.
parameters:
  - attr: <class 'str'> = weight --The edge attribute used to in determining optimality.
  - default: <class 'float'> = 1 --The value of the edge attribute used if an edge does not have
the attribute `attr`.
  - preserve_attrs: <class 'bool'> = ? --If True, preserve the other attributes of the original graph (that are not
passed to `attr`)
  - partition: <class 'str'> = ? --The key for the edge attribute containing the partition
data on the graph. Edges can be included, excluded or open using the
`EdgePartition` enum.
  - G: <class 'networkx.classes.graph.Graph'> = ? --The graph to be searched.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.tree.branchings.minimum_spanning_arborescence(attr=<attr_value>, default=<default_value>, preserve_attrs=<preserve_attrs_value>, partition=<partition_value>, G=<G_variable>)
