**Could be isomorphic:**
Returns False if graphs are definitely not isomorphic.
True does NOT guarantee isomorphism.
parameters:
  - G1: <class 'networkx.classes.graph.Graph'> = ? --The two graphs `G1` and `G2` must be the same type.
  - G2: <class 'networkx.classes.graph.Graph'> = ? --The two graphs `G1` and `G2` must be the same type.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.isomorphism.isomorph.could_be_isomorphic(G1=<G1_variable>, G2=<G2_variable>)
