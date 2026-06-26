**VF2++ all isomorphisms:**
Yields all the possible mappings between G1 and G2.
parameters:
  - node_label: str | None = ? --The name of the node attribute to be used when comparing nodes.
The default is `None`, meaning node attributes are not considered
in the comparison. Any node that doesn't have the `node_label`
attribute uses `default_label` instead.
  - default_label: <class 'float'> = ? --Default value to use when a node doesn't have an attribute
named `node_label`. Default is `None`.
  - G1: <class 'networkx.classes.graph.Graph'> = ? --The two graphs to check for isomorphism.
  - G2: <class 'networkx.classes.graph.Graph'> = ? --The two graphs to check for isomorphism.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.isomorphism.vf2pp.vf2pp_all_isomorphisms(node_label=<node_label_value>, default_label=<default_label_value>, G1=<G1_variable>, G2=<G2_variable>)
