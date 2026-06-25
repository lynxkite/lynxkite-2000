**Is connected:**
Returns True if the graph is connected, False otherwise.

A graph is connected if, for every pair of distinct nodes, there is a
path between them. If there is a pair of nodes for which such path does
not exist, the graph is not connected (also referred to as "disconnected").

A graph consisting of a single node and no edges is connected.
Connectivity is undefined for the null graph (graph with no nodes).
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --An undirected graph.
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.components.connected.is_connected(G=<G_variable>)
