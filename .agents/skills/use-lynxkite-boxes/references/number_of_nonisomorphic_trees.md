**Number of nonisomorphic trees:**
Returns the number of nonisomorphic trees of the specified `order`.

Based on an algorithm by Alois P. Heinz in
`OEIS entry A000055 <https://oeis.org/A000055>`_. Complexity is ``O(n ** 3)``.
parameters:
  - order: <class 'int'> = ? --Order of the desired tree(s).
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.generators.nonisomorphic_trees.number_of_nonisomorphic_trees(order=<order_value>)
