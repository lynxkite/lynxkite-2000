---
name: networkx-generators-mycielski
description: Collection of operations - Mycielskian, Mycielski graph
---

**Mycielskian:**
Returns the Mycielskian of a simple, undirected graph G

The Mycielskian of graph preserves a graph's triangle free
property while increasing the chromatic number by 1.

The Mycielski Operation on a graph, :math:`G=(V, E)`, constructs a new
graph with :math:`2|V| + 1` nodes and :math:`3|E| + |V|` edges.

The construction is as follows:

Let :math:`V = {0, ..., n-1}`. Construct another vertex set
:math:`U = {n, ..., 2n}` and a vertex, `w`.
Construct a new graph, `M`, with vertices :math:`U \bigcup V \bigcup w`.
For edges, :math:`(u, v) \in E` add edges :math:`(u, v), (u, v + n)`, and
:math:`(u + n, v)` to M. Finally, for all vertices :math:`u \in U`, add
edge :math:`(u, w)` to M.

The Mycielski Operation can be done multiple times by repeating the above
process iteratively.

More information can be found at https://en.wikipedia.org/wiki/Mycielskian
parameters:
  - iterations: <class 'int'> = 1 -
  - G: <class 'networkx.classes.graph.Graph'> = None -

usage:
output_variable = networkx.generators.mycielski.mycielskian(iterations=<iterations_value>, G=<G_variable>)

**Mycielski graph:**
Generator for the n_th Mycielski Graph.

The Mycielski family of graphs is an infinite set of graphs.
:math:`M_1` is the singleton graph, :math:`M_2` is two vertices with an
edge, and, for :math:`i > 2`, :math:`M_i` is the Mycielskian of
:math:`M_{i-1}`.

More information can be found at
http://mathworld.wolfram.com/MycielskiGraph.html
parameters:
  - n: <class 'int'> = None -

usage:
output_variable = networkx.generators.mycielski.mycielski_graph(n=<n_value>)
