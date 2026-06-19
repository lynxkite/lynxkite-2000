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
