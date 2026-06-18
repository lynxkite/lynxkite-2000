---
name: networkx-algorithms-isomorphism-vf2pp
description: Collection of operations - VF2++ isomorphism, VF2++ is isomorphic, VF2++ all isomorphisms
---

**VF2++ isomorphism:**
Return an isomorphic mapping between `G1` and `G2` if it exists.
parameters:
  - node_label: str | None = None -
  - default_label: <class 'float'> = None -
  - G1: <class 'networkx.classes.graph.Graph'> = None -
  - G2: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.isomorphism.vf2pp.vf2pp_isomorphism(node_label=<node_label_value>, default_label=<default_label_value>, G1=<G1_variable>, G2=<G2_variable>)

**VF2++ is isomorphic:**
Examines whether G1 and G2 are isomorphic.
parameters:
  - node_label: str | None = None -
  - default_label: <class 'float'> = None -
  - G1: <class 'networkx.classes.graph.Graph'> = None -
  - G2: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.isomorphism.vf2pp.vf2pp_is_isomorphic(node_label=<node_label_value>, default_label=<default_label_value>, G1=<G1_variable>, G2=<G2_variable>)

**VF2++ all isomorphisms:**
Yields all the possible mappings between G1 and G2.
parameters:
  - node_label: str | None = None -
  - default_label: <class 'float'> = None -
  - G1: <class 'networkx.classes.graph.Graph'> = None -
  - G2: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.isomorphism.vf2pp.vf2pp_all_isomorphisms(node_label=<node_label_value>, default_label=<default_label_value>, G1=<G1_variable>, G2=<G2_variable>)
