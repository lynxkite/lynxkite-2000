---
name: networkx-algorithms-summarization
description: Collection of operations - Dedensify, Snap aggregation
---

**Dedensify:**
Compresses neighborhoods around high-degree nodes

Reduces the number of edges to high-degree nodes by adding compressor nodes
that summarize multiple edges of the same type to high-degree nodes (nodes
with a degree greater than a given threshold).  Dedensification also has
the added benefit of reducing the number of edges around high-degree nodes.
The implementation currently supports graphs with a single edge type.
parameters:
  - threshold: <class 'int'> = ? --Minimum degree threshold of a node to be considered a high degree node.
The threshold must be greater than or equal to 2.
  - copy: bool | None = ? --Indicates if dedensification should be done inplace
  - G: <class 'networkx.classes.graph.Graph'> = ? --A networkx graph

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.summarization.dedensify(threshold=<threshold_value>, copy=<copy_value>, G=<G_variable>)

**Snap aggregation:**
Creates a summary graph based on attributes and connectivity.

This function uses the Summarization by Grouping Nodes on Attributes
and Pairwise edges (SNAP) algorithm for summarizing a given
graph by grouping nodes by node attributes and their edge attributes
into supernodes in a summary graph.  This name SNAP should not be
confused with the Stanford Network Analysis Project (SNAP).

Here is a high-level view of how this algorithm works:

1) Group nodes by node attribute values.

2) Iteratively split groups until all nodes in each group have edges
to nodes in the same groups. That is, until all the groups are homogeneous
in their member nodes' edges to other groups.  For example,
if all the nodes in group A only have edge to nodes in group B, then the
group is homogeneous and does not need to be split. If all nodes in group B
have edges with nodes in groups {A, C}, but some also have edges with other
nodes in B, then group B is not homogeneous and needs to be split into
groups have edges with {A, C} and a group of nodes having
edges with {A, B, C}.  This way, viewers of the summary graph can
assume that all nodes in the group have the exact same node attributes and
the exact same edges.

3) Build the output summary graph, where the groups are represented by
super-nodes. Edges represent the edges shared between all the nodes in each
respective groups.

A SNAP summary graph can be used to visualize graphs that are too large to display
or visually analyze, or to efficiently identify sets of similar nodes with similar connectivity
patterns to other sets of similar nodes based on specified node and/or edge attributes in a graph.
parameters:
  - prefix: <class 'str'> = Supernode- --The prefix used to denote supernodes in the summary graph. Defaults to 'Supernode-'.
  - supernode_attribute: <class 'str'> = group --The node attribute for recording the supernode groupings of nodes. Defaults to 'group'.
  - superedge_attribute: <class 'str'> = types --The edge attribute for recording the edge types of multiple edges. Defaults to 'types'.
  - G: <class 'networkx.classes.graph.Graph'> = ? --Networkx Graph to be summarized

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.summarization.snap_aggregation(prefix=<prefix_value>, supernode_attribute=<supernode_attribute_value>, superedge_attribute=<superedge_attribute_value>, G=<G_variable>)
