---
name: is-semiconnected
description: Returns True if the graph is semiconnected, False otherwise.
---

**Is semiconnected:**
Returns True if the graph is semiconnected, False otherwise.

A graph is semiconnected if and only if for any pair of nodes, either one
is reachable from the other, or they are mutually reachable.

This function uses a theorem that states that a DAG is semiconnected
if for any topological sort, for node $v_n$ in that sort, there is an
edge $(v_i, v_{i+1})$. That allows us to check if a non-DAG `G` is
semiconnected by condensing the graph: i.e. constructing a new graph `H`
with nodes being the strongly connected components of `G`, and edges
(scc_1, scc_2) if there is a edge $(v_1, v_2)$ in `G` for some
$v_1 \in scc_1$ and $v_2 \in scc_2$. That results in a DAG, so we compute
the topological sort of `H` and check if for every $n$ there is an edge
$(scc_n, scc_{n+1})$.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --A directed graph.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.components.semiconnected.is_semiconnected(G=<G_variable>)
