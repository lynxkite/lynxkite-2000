**Junction tree:**
Returns a junction tree of a given graph.

A junction tree (or clique tree) is constructed from a (un)directed graph G.
The tree is constructed based on a moralized and triangulated version of G.
The tree's nodes consist of maximal cliques and sepsets of the revised graph.
The sepset of two cliques is the intersection of the nodes of these cliques,
e.g. the sepset of (A,B,C) and (A,C,E,F) is (A,C). These nodes are often called
"variables" in this literature. The tree is bipartite with each sepset
connected to its two cliques.

Junction Trees are not unique as the order of clique consideration determines
which sepsets are included.

The junction tree algorithm consists of five steps [1]_:

1. Moralize the graph
2. Triangulate the graph
3. Find maximal cliques
4. Build the tree from cliques, connecting cliques with shared
   nodes, set edge-weight to number of shared variables
5. Find maximum spanning tree
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --Directed or undirected graph.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.tree.decomposition.junction_tree(G=<G_variable>)
