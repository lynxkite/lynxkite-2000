---
name: networkx-algorithms-isomorphism-isomorph
description: Collection of operations - Could be isomorphic, Fast could be isomorphic, Faster could be isomorphic
---

**Could be isomorphic:**
Returns False if graphs are definitely not isomorphic.
True does NOT guarantee isomorphism.
parameters:
  - G1: <class 'networkx.classes.graph.Graph'> = None - .
  - G2: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.algorithms.isomorphism.isomorph.could_be_isomorphic(G1=<G1_variable>, G2=<G2_variable>)

**Fast could be isomorphic:**
Returns False if graphs are definitely not isomorphic.

True does NOT guarantee isomorphism.
parameters:
  - G1: <class 'networkx.classes.graph.Graph'> = None - .
  - G2: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.algorithms.isomorphism.isomorph.fast_could_be_isomorphic(G1=<G1_variable>, G2=<G2_variable>)

**Faster could be isomorphic:**
Returns False if graphs are definitely not isomorphic.

True does NOT guarantee isomorphism.
parameters:
  - G1: <class 'networkx.classes.graph.Graph'> = None - .
  - G2: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.algorithms.isomorphism.isomorph.faster_could_be_isomorphic(G1=<G1_variable>, G2=<G2_variable>)
