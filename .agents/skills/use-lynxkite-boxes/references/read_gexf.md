**Read gexf:**
Read graph in GEXF format from path.

"GEXF (Graph Exchange XML Format) is a language for describing
complex networks structures, their associated data and dynamics" [1]_.
parameters:
  - relabel: <class 'bool'> = ? --If True relabel the nodes to use the GEXF node "label" attribute
instead of the node "id" attribute as the NetworkX node label.
  - version: <class 'str'> = 1.2draft --?

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.readwrite.gexf.read_gexf(relabel=<relabel_value>, version=<version_value>)
