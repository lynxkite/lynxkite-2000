---
name: network-simplex
description: Find a minimum cost flow satisfying all demands in digraph G.
---

**Network simplex:**
Find a minimum cost flow satisfying all demands in digraph G.

This is a primal network simplex algorithm that uses the leaving
arc rule to prevent cycling.

G is a digraph with edge costs and capacities and in which nodes
have demand, i.e., they want to send or receive some amount of
flow. A negative demand means that the node wants to send flow, a
positive demand means that the node want to receive flow. A flow on
the digraph G satisfies all demand if the net flow into each node
is equal to the demand of that node.
parameters:
  - demand: <class 'str'> = demand -
  - capacity: <class 'str'> = capacity -
  - weight: <class 'str'> = weight -
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.algorithms.flow.networksimplex.network_simplex(demand=<demand_value>, capacity=<capacity_value>, weight=<weight_value>, G=<G_variable>)
