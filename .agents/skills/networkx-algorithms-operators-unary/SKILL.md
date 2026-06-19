---
name: networkx-algorithms-operators-unary
description: Collection of operations - Complement, Reverse
---

**Complement:**
Returns the graph complement of G.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --A NetworkX graph

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.operators.unary.complement(G=<G_variable>)

**Reverse:**
Returns the reverse directed graph of G.
parameters:
  - copy: <class 'bool'> = ? --If True, then a new graph is returned. If False, then the graph is
reversed in place.
  - G: <class 'networkx.classes.graph.Graph'> = ? --A NetworkX directed graph

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.operators.unary.reverse(copy=<copy_value>, G=<G_variable>)
