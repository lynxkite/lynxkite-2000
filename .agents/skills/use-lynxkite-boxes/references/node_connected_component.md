**Node connected component:**
Returns the set of nodes in the component of graph containing node n.

A connected component is a set of nodes that induces a subgraph of graph
`G` that is connected and not part of any larger connected subgraph.

A graph is connected (:func:`is_connected`) if, for every pair of distinct
nodes, there is a path between them. If there is a pair of nodes for
which such path does not exist, the graph is not connected (also referred
to as "disconnected").

A graph consisting of a single node and no edges is connected.
Connectivity is undefined for the null graph (graph with no nodes).
parameters:
  - n: <class 'str'> = ? --A node in G
  - G: <class 'networkx.classes.graph.Graph'> = ? --An undirected graph.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.components.connected.node_connected_component(n=<n_value>, G=<G_variable>)
