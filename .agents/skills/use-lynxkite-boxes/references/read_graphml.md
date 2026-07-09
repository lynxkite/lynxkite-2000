**Read GraphML:**
Read graph in GraphML format from path.
parameters:
  - force_multigraph: <class 'bool'> = ? --If True, return a multigraph with edge keys. If False (the default)
return a multigraph when multiedges are in the graph.
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.readwrite.graphml.read_graphml(force_multigraph=<force_multigraph_value>)
