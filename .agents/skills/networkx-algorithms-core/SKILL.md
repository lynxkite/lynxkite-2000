---
name: networkx-algorithms-core
description: Collection of operations - Core number, k-core, k-shell, k-crust, k-corona, k-truss, Onion layers
---

**Core number:**
Returns the core number for each node.

A k-core is a maximal subgraph that contains nodes of degree k or more.

The core number of a node is the largest value k of a k-core containing
that node.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --An undirected or directed graph

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.core.core_number(G=<G_variable>)

**k-core:**
Returns the k-core of G.

A k-core is a maximal subgraph that contains nodes of degree `k` or more.
parameters:
  - k: int | None = ? --The order of the core. If not specified return the main core.
  - G: <class 'networkx.classes.graph.Graph'> = ? --A graph or directed graph

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.core.k_core(k=<k_value>, G=<G_variable>)

**k-shell:**
Returns the k-shell of G.

The k-shell is the subgraph induced by nodes with core number k.
That is, nodes in the k-core that are not in the (k+1)-core.
parameters:
  - k: int | None = ? --The order of the shell. If not specified return the outer shell.
  - G: <class 'networkx.classes.graph.Graph'> = ? --A graph or directed graph.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.core.k_shell(k=<k_value>, G=<G_variable>)

**k-crust:**
Returns the k-crust of G.

The k-crust is the graph G with the edges of the k-core removed
and isolated nodes found after the removal of edges are also removed.
parameters:
  - k: int | None = ? --The order of the shell. If not specified return the main crust.
  - G: <class 'networkx.classes.graph.Graph'> = ? --A graph or directed graph.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.core.k_crust(k=<k_value>, G=<G_variable>)

**k-corona:**
Returns the k-corona of G.

The k-corona is the subgraph of nodes in the k-core which have
exactly k neighbors in the k-core.
parameters:
  - k: <class 'int'> = ? --The order of the corona.
  - G: <class 'networkx.classes.graph.Graph'> = ? --A graph or directed graph

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.core.k_corona(k=<k_value>, G=<G_variable>)

**k-truss:**
Returns the k-truss of `G`.

The k-truss is the maximal induced subgraph of `G` which contains at least
three vertices where every edge is incident to at least `k-2` triangles.
parameters:
  - k: <class 'int'> = ? --The order of the truss
  - G: <class 'networkx.classes.graph.Graph'> = ? --An undirected graph

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.core.k_truss(k=<k_value>, G=<G_variable>)

**Onion layers:**
Returns the layer of each vertex in an onion decomposition of the graph.

The onion decomposition refines the k-core decomposition by providing
information on the internal organization of each k-shell. It is usually
used alongside the `core numbers`.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --An undirected graph without self loops.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.core.onion_layers(G=<G_variable>)
