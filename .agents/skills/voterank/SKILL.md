---
name: voterank
description: Select a list of influential nodes in a graph using VoteRank algorithm
---

**Voterank:**
Select a list of influential nodes in a graph using VoteRank algorithm

VoteRank [1]_ computes a ranking of the nodes in a graph G based on a
voting scheme. With VoteRank, all nodes vote for each of its in-neighbors
and the node with the highest votes is elected iteratively. The voting
ability of out-neighbors of elected nodes is decreased in subsequent turns.
parameters:
  - number_of_nodes: int | None = None - .
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.algorithms.centrality.voterank_alg.voterank(number_of_nodes=<number_of_nodes_value>, G=<G_variable>)
