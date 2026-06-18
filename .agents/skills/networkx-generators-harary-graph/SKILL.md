---
name: networkx-generators-harary-graph
description: Collection of operations - H(n,m) Harary graph, H(k,n) Harary graph
---

**H(n,m) Harary graph:**
Return the Harary graph with given numbers of nodes and edges.

The Harary graph $H_{n, m}$ is the graph that maximizes node connectivity
with $n$ nodes and $m$ edges.

This maximum node connectivity is known to be $\lfloor 2m/n \rfloor$. [1]_
parameters:
  - n: <class 'int'> = None -
  - m: <class 'int'> = None -

usage:
output_variable = networkx.generators.harary_graph.hnm_harary_graph(n=<n_value>, m=<m_value>)

**H(k,n) Harary graph:**
Return the Harary graph with given node connectivity and node number.

The Harary graph $H_{k, n}$ is the graph that minimizes the number of
edges needed with given node connectivity $k$ and node number $n$.

This smallest number of edges is known to be $\lceil kn/2 \rceil$ [1]_.
parameters:
  - k: <class 'int'> = None -
  - n: <class 'int'> = None -

usage:
output_variable = networkx.generators.harary_graph.hkn_harary_graph(k=<k_value>, n=<n_value>)
