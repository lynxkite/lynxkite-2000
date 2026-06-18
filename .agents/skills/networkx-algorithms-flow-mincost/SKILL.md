---
name: networkx-algorithms-flow-mincost
description: Collection of operations - Max flow min cost, Min cost flow, Min cost flow cost
---

**Max flow min cost:**
Returns a maximum (s, t)-flow of minimum cost.

G is a digraph with edge costs and capacities. There is a source
node s and a sink node t. This function finds a maximum flow from
s to t whose total cost is minimized.
parameters:
  - s: <class 'str'> = None - .
  - t: <class 'str'> = None - .
  - capacity: <class 'str'> = capacity - .
  - weight: <class 'str'> = weight - .
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.algorithms.flow.mincost.max_flow_min_cost(s=<s_value>, t=<t_value>, capacity=<capacity_value>, weight=<weight_value>, G=<G_variable>)

**Min cost flow:**
Returns a minimum cost flow satisfying all demands in digraph G.

G is a digraph with edge costs and capacities and in which nodes
have demand, i.e., they want to send or receive some amount of
flow. A negative demand means that the node wants to send flow, a
positive demand means that the node want to receive flow. A flow on
the digraph G satisfies all demand if the net flow into each node
is equal to the demand of that node.
parameters:
  - demand: <class 'str'> = demand - .
  - capacity: <class 'str'> = capacity - .
  - weight: <class 'str'> = weight - .
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.algorithms.flow.mincost.min_cost_flow(demand=<demand_value>, capacity=<capacity_value>, weight=<weight_value>, G=<G_variable>)

**Min cost flow cost:**
Find the cost of a minimum cost flow satisfying all demands in digraph G.

G is a digraph with edge costs and capacities and in which nodes
have demand, i.e., they want to send or receive some amount of
flow. A negative demand means that the node wants to send flow, a
positive demand means that the node want to receive flow. A flow on
the digraph G satisfies all demand if the net flow into each node
is equal to the demand of that node.
parameters:
  - demand: <class 'str'> = demand - .
  - capacity: <class 'str'> = capacity - .
  - weight: <class 'str'> = weight - .
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.algorithms.flow.mincost.min_cost_flow_cost(demand=<demand_value>, capacity=<capacity_value>, weight=<weight_value>, G=<G_variable>)
