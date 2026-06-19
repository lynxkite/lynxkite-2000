---
name: networkx-generators-trees
description: Collection of operations - Prefix tree, Prefix tree recursive, Random labeled tree, Random labeled rooted tree, Random labeled rooted forest, Random unlabeled tree, Random unlabeled rooted tree, Random unlabeled rooted forest
---

**Prefix tree:**
Creates a directed prefix tree from a list of paths.

Usually the paths are described as strings or lists of integers.

A "prefix tree" represents the prefix structure of the strings.
Each node represents a prefix of some string. The root represents
the empty prefix with children for the single letter prefixes which
in turn have children for each double letter prefix starting with
the single letter corresponding to the parent node, and so on.

More generally the prefixes do not need to be strings. A prefix refers
to the start of a sequence. The root has children for each one element
prefix and they have children for each two element prefix that starts
with the one element sequence of the parent, and so on.

Note that this implementation uses integer nodes with an attribute.
Each node has an attribute "source" whose value is the original element
of the path to which this node corresponds. For example, suppose `paths`
consists of one path: "can". Then the nodes `[1, 2, 3]` which represent
this path have "source" values "c", "a" and "n".

All the descendants of a node have a common prefix in the sequence/path
associated with that node. From the returned tree, the prefix for each
node can be constructed by traversing the tree up to the root and
accumulating the "source" values along the way.

The root node is always `0` and has "source" attribute `None`.
The root is the only node with in-degree zero.
The nil node is always `-1` and has "source" attribute `"NIL"`.
The nil node is the only node with out-degree zero.
parameters:


returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.trees.prefix_tree()

**Prefix tree recursive:**
Recursively creates a directed prefix tree from a list of paths.

The original recursive version of prefix_tree for comparison. It is
the same algorithm but the recursion is unrolled onto a stack.

Usually the paths are described as strings or lists of integers.

A "prefix tree" represents the prefix structure of the strings.
Each node represents a prefix of some string. The root represents
the empty prefix with children for the single letter prefixes which
in turn have children for each double letter prefix starting with
the single letter corresponding to the parent node, and so on.

More generally the prefixes do not need to be strings. A prefix refers
to the start of a sequence. The root has children for each one element
prefix and they have children for each two element prefix that starts
with the one element sequence of the parent, and so on.

Note that this implementation uses integer nodes with an attribute.
Each node has an attribute "source" whose value is the original element
of the path to which this node corresponds. For example, suppose `paths`
consists of one path: "can". Then the nodes `[1, 2, 3]` which represent
this path have "source" values "c", "a" and "n".

All the descendants of a node have a common prefix in the sequence/path
associated with that node. From the returned tree, ehe prefix for each
node can be constructed by traversing the tree up to the root and
accumulating the "source" values along the way.

The root node is always `0` and has "source" attribute `None`.
The root is the only node with in-degree zero.
The nil node is always `-1` and has "source" attribute `"NIL"`.
The nil node is the only node with out-degree zero.
parameters:


returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.trees.prefix_tree_recursive()

**Random labeled tree:**
Returns a labeled tree on `n` nodes chosen uniformly at random.

Generating uniformly distributed random Prüfer sequences and
converting them into the corresponding trees is a straightforward
method of generating uniformly distributed random labeled trees.
This function implements this method.
parameters:
  - n: <class 'int'> = ? --The number of nodes, greater than zero.
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.trees.random_labeled_tree(n=<n_value>, seed=<seed_value>)

**Random labeled rooted tree:**
Returns a labeled rooted tree with `n` nodes.

The returned tree is chosen uniformly at random from all labeled rooted trees.
parameters:
  - n: <class 'int'> = ? --The number of nodes
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.trees.random_labeled_rooted_tree(n=<n_value>, seed=<seed_value>)

**Random labeled rooted forest:**
Returns a labeled rooted forest with `n` nodes.

The returned forest is chosen uniformly at random using a
generalization of Prüfer sequences [1]_ in the form described in [2]_.
parameters:
  - n: <class 'int'> = ? --The number of nodes.
  - seed: int | None = ? --See :ref:`Randomness<randomness>`.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.trees.random_labeled_rooted_forest(n=<n_value>, seed=<seed_value>)

**Random unlabeled tree:**
Returns a tree or list of trees chosen randomly.

Returns one or more (depending on `number_of_trees`)
unlabeled trees with `n` nodes drawn uniformly at random.
parameters:
  - n: <class 'int'> = ? --The number of nodes
  - number_of_trees: int | None = ? --If not None, this number of trees is generated and returned.
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.trees.random_unlabeled_tree(n=<n_value>, number_of_trees=<number_of_trees_value>, seed=<seed_value>)

**Random unlabeled rooted tree:**
Returns a number of unlabeled rooted trees uniformly at random

Returns one or more (depending on `number_of_trees`)
unlabeled rooted trees with `n` nodes drawn uniformly
at random.
parameters:
  - n: <class 'int'> = ? --The number of nodes
  - number_of_trees: int | None = ? --If not None, this number of trees is generated and returned.
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.trees.random_unlabeled_rooted_tree(n=<n_value>, number_of_trees=<number_of_trees_value>, seed=<seed_value>)

**Random unlabeled rooted forest:**
Returns a forest or list of forests selected at random.

Returns one or more (depending on `number_of_forests`)
unlabeled rooted forests with `n` nodes, and with no more than
`q` nodes per tree, drawn uniformly at random.
The "roots" graph attribute identifies the roots of the forest.
parameters:
  - n: <class 'int'> = ? --The number of nodes
  - q: int | None = ? --The maximum number of nodes per tree.
  - number_of_forests: int | None = ? --If not None, this number of forests is generated and returned.
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.trees.random_unlabeled_rooted_forest(n=<n_value>, q=<q_value>, number_of_forests=<number_of_forests_value>, seed=<seed_value>)
