---
name: networkx-generators-nonisomorphic-trees
description: Collection of operations - Nonisomorphic trees, Number of nonisomorphic trees
---

**Nonisomorphic trees:**
Generate nonisomorphic trees of specified `order`.
parameters:
  - order: <class 'int'> = None - .

usage:
output_variable = networkx.generators.nonisomorphic_trees.nonisomorphic_trees(order=<order_value>)

**Number of nonisomorphic trees:**
Returns the number of nonisomorphic trees of the specified `order`.

Based on an algorithm by Alois P. Heinz in
`OEIS entry A000055 <https://oeis.org/A000055>`_. Complexity is ``O(n ** 3)``.
parameters:
  - order: <class 'int'> = None - .

usage:
output_variable = networkx.generators.nonisomorphic_trees.number_of_nonisomorphic_trees(order=<order_value>)
