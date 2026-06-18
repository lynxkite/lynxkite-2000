---
name: networkx-algorithms-operators-unary
description: Collection of operations - Complement, Reverse
---

**Complement:**
Returns the graph complement of G.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.algorithms.operators.unary.complement(G=<G_variable>)

**Reverse:**
Returns the reverse directed graph of G.
parameters:
  - copy: <class 'bool'> = None - .
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.algorithms.operators.unary.reverse(copy=<copy_value>, G=<G_variable>)
