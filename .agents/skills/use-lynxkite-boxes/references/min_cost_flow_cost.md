**Min cost flow cost:**
Find the cost of a minimum cost flow satisfying all demands in digraph G.

G is a digraph with edge costs and capacities and in which nodes
have demand, i.e., they want to send or receive some amount of
flow. A negative demand means that the node wants to send flow, a
positive demand means that the node want to receive flow. A flow on
the digraph G satisfies all demand if the net flow into each node
is equal to the demand of that node.
parameters:
  - demand: <class 'str'> = demand --Nodes of the graph G are expected to have an attribute demand
that indicates how much flow a node wants to send (negative
demand) or receive (positive demand). Note that the sum of the
demands should be 0 otherwise the problem in not feasible. If
this attribute is not present, a node is considered to have 0
demand. Default value: 'demand'.
  - capacity: <class 'str'> = capacity --Edges of the graph G are expected to have an attribute capacity
that indicates how much flow the edge can support. If this
attribute is not present, the edge is considered to have
infinite capacity. Default value: 'capacity'.
  - weight: <class 'str'> = weight --Edges of the graph G are expected to have an attribute weight
that indicates the cost incurred by sending one unit of flow on
that edge. If not present, the weight is considered to be 0.
Default value: 'weight'.
  - G: <class 'networkx.classes.graph.Graph'> = ? --DiGraph on which a minimum cost flow satisfying all demands is
to be found.
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.flow.mincost.min_cost_flow_cost(demand=<demand_value>, capacity=<capacity_value>, weight=<weight_value>, G=<G_variable>)
