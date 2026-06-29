**Max flow min cost:**
Returns a maximum (s, t)-flow of minimum cost.

G is a digraph with edge costs and capacities. There is a source
node s and a sink node t. This function finds a maximum flow from
s to t whose total cost is minimized.
parameters:
  - s: <class 'str'> = ? --Source of the flow.
  - t: <class 'str'> = ? --Destination of the flow.
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
output_variable = networkx.algorithms.flow.mincost.max_flow_min_cost(s=<s_value>, t=<t_value>, capacity=<capacity_value>, weight=<weight_value>, G=<G_variable>)
