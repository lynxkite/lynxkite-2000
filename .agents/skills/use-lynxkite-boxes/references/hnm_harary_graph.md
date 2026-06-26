**H(n,m) Harary graph:**
Return the Harary graph with given numbers of nodes and edges.

The Harary graph $H_{n, m}$ is the graph that maximizes node connectivity
with $n$ nodes and $m$ edges.

This maximum node connectivity is known to be $\lfloor 2m/n \rfloor$. [1]_
parameters:
  - n: <class 'int'> = ? --The number of nodes the generated graph is to contain.
  - m: <class 'int'> = ? --The number of edges the generated graph is to contain.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.harary_graph.hnm_harary_graph(n=<n_value>, m=<m_value>)
