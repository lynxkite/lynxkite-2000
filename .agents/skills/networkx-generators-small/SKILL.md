---
name: networkx-generators-small
description: Collection of operations - Bull graph, Chvatal graph, Cubical graph, Desargues graph, Diamond graph, Dodecahedral graph, Frucht graph, Generalized petersen graph, Heawood graph, Hoffman singleton graph, House graph, House x graph, Icosahedral graph, Krackhardt kite graph, Moebius–Kantor graph, Octahedral graph, Pappus graph, Petersen graph, Sedgewick maze graph, Tetrahedral graph, Truncated cube graph, Truncated tetrahedron graph, Tutte graph
---

**Bull graph:**
Returns the Bull Graph

The Bull Graph has 5 nodes and 5 edges. It is a planar undirected
graph in the form of a triangle with two disjoint pendant edges [1]_
The name comes from the triangle and pendant edges representing
respectively the body and legs of a bull.
parameters:


returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.small.bull_graph()

**Chvatal graph:**
Returns the Chvátal Graph

The Chvátal Graph is an undirected graph with 12 nodes and 24 edges [1]_.
It has 370 distinct (directed) Hamiltonian cycles, giving a unique generalized
LCF notation of order 4, two of order 6 , and 43 of order 1 [2]_.
parameters:


returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.small.chvatal_graph()

**Cubical graph:**
Returns the 3-regular Platonic Cubical Graph

The skeleton of the cube (the nodes and edges) form a graph, with 8
nodes, and 12 edges. It is a special case of the hypercube graph.
It is one of 5 Platonic graphs, each a skeleton of its
Platonic solid [1]_.
Such graphs arise in parallel processing in computers.
parameters:


returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.small.cubical_graph()

**Desargues graph:**
Returns the Desargues Graph

The Desargues Graph is a non-planar, distance-transitive cubic graph
with 20 nodes and 30 edges [1]_. It is isomorphic to the Generalized
Petersen Graph GP(10, 3). It is a symmetric graph. It can be represented
in LCF notation as [5,-5,9,-9]^5 [2]_.
parameters:


returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.small.desargues_graph()

**Diamond graph:**
Returns the Diamond graph

The Diamond Graph is  planar undirected graph with 4 nodes and 5 edges.
It is also sometimes known as the double triangle graph or kite graph [1]_.
parameters:


returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.small.diamond_graph()

**Dodecahedral graph:**
Returns the Platonic Dodecahedral graph.

The dodecahedral graph has 20 nodes and 30 edges. The skeleton of the
dodecahedron forms a graph. It is one of 5 Platonic graphs [1]_.
It can be described in LCF notation as:
``[10, 7, 4, -4, -7, 10, -4, 7, -7, 4]^2`` [2]_.
parameters:


returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.small.dodecahedral_graph()

**Frucht graph:**
Returns the Frucht Graph.

The Frucht Graph is the smallest cubical graph whose
automorphism group consists only of the identity element [1]_.
It has 12 nodes and 18 edges and no nontrivial symmetries.
It is planar and Hamiltonian [2]_.
parameters:


returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.small.frucht_graph()

**Generalized petersen graph:**
Returns the Generalized Petersen Graph GP(n,k).

The Generalized Peterson Graph consists of an outer cycle of n nodes
connected to an inner circulant graph of n nodes, where nodes in the
inner circulant are connected to their kth nearest neighbor [1]_ [2]_.
A Generalized Petersen Graph is cubic with 2n nodes and 3n edges.

Some well known graphs are examples of Generalized Petersen Graphs such
as the Petersen Graph GP(5, 2), the Desargues graph GP(10, 3), the
Moebius-Kantor graph GP(8, 3), and the dodecahedron graph GP(10, 2).
parameters:
  - n: <class 'int'> = ? --Number of nodes in the outer cycle and inner circulant. ``n >= 3`` is required.
  - k: <class 'int'> = ? --Neighbor to connect in the inner circulant. ``1 <= k <= n/2``.
Note that some people require ``k < n/2`` but we and others allow equality.
Also, ``k < n/2`` is equivalent to ``k <= floor((n-1)/2)``

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.small.generalized_petersen_graph(n=<n_value>, k=<k_value>)

**Heawood graph:**
Returns the Heawood Graph, a (3,6) cage.

The Heawood Graph is an undirected graph with 14 nodes and 21 edges,
named after Percy John Heawood [1]_.
It is cubic symmetric, nonplanar, Hamiltonian, and can be represented
in LCF notation as ``[5,-5]^7`` [2]_.
It is the unique (3,6)-cage: the regular cubic graph of girth 6 with
minimal number of vertices [3]_.
parameters:


returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.small.heawood_graph()

**Hoffman singleton graph:**
Returns the Hoffman-Singleton Graph.

The Hoffman–Singleton graph is a symmetrical undirected graph
with 50 nodes and 175 edges.
All indices lie in ``Z % 5``: that is, the integers mod 5 [1]_.
It is the only regular graph of vertex degree 7, diameter 2, and girth 5.
It is the unique (7,5)-cage graph and Moore graph, and contains many
copies of the Petersen Graph [2]_.
parameters:


returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.small.hoffman_singleton_graph()

**House graph:**
Returns the House graph (square with triangle on top)

The house graph is a simple undirected graph with
5 nodes and 6 edges [1]_.
parameters:


returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.small.house_graph()

**House x graph:**
Returns the House graph with a cross inside the house square.

The House X-graph is the House graph plus the two edges connecting diagonally
opposite vertices of the square base. It is also one of the two graphs
obtained by removing two edges from the pentatope graph [1]_.
parameters:


returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.small.house_x_graph()

**Icosahedral graph:**
Returns the Platonic Icosahedral graph.

The icosahedral graph has 12 nodes and 30 edges. It is a Platonic graph
whose nodes have the connectivity of the icosahedron. It is undirected,
regular and Hamiltonian [1]_.
parameters:


returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.small.icosahedral_graph()

**Krackhardt kite graph:**
Returns the Krackhardt Kite Social Network.

A 10 actor social network introduced by David Krackhardt
to illustrate different centrality measures [1]_.
parameters:


returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.small.krackhardt_kite_graph()

**Moebius–Kantor graph:**
Returns the Moebius-Kantor graph.

The Möbius-Kantor graph is the cubic symmetric graph on 16 nodes.
Its LCF notation is [5,-5]^8, and it is isomorphic to the generalized
Petersen Graph GP(8, 3) [1]_.
parameters:


returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.small.moebius_kantor_graph()

**Octahedral graph:**
Returns the Platonic Octahedral graph.

The octahedral graph is the 6-node 12-edge Platonic graph having the
connectivity of the octahedron [1]_. If 6 couples go to a party,
and each person shakes hands with every person except his or her partner,
then this graph describes the set of handshakes that take place;
for this reason it is also called the cocktail party graph [2]_.
parameters:


returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.small.octahedral_graph()

**Pappus graph:**
Returns the Pappus graph.

The Pappus graph is a cubic symmetric distance-regular graph with 18 nodes
and 27 edges. It is Hamiltonian and can be represented in LCF notation as
[5,7,-7,7,-7,-5]^3 [1]_.
parameters:


returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.small.pappus_graph()

**Petersen graph:**
Returns the Petersen Graph.

The Peterson Graph is a cubic, undirected graph with 10 nodes and 15 edges [1]_.
Julius Petersen constructed the graph as the smallest counterexample
against the claim that a connected bridgeless cubic graph
has an edge colouring with three colours [2]_.
parameters:


returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.small.petersen_graph()

**Sedgewick maze graph:**
Return a small maze with a cycle.

This is the maze used in Sedgewick, 3rd Edition, Part 5, Graph
Algorithms, Chapter 18, e.g. Figure 18.2 and following [1]_.
Nodes are numbered 0,..,7
parameters:


returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.small.sedgewick_maze_graph()

**Tetrahedral graph:**
Returns the 3-regular Platonic Tetrahedral graph.

Tetrahedral graph has 4 nodes and 6 edges. It is a
special case of the complete graph, K4, and wheel graph, W4.
It is one of the 5 platonic graphs [1]_.
parameters:


returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.small.tetrahedral_graph()

**Truncated cube graph:**
Returns the skeleton of the truncated cube.

The truncated cube is an Archimedean solid with 14 regular
faces (6 octagonal and 8 triangular), 36 edges and 24 nodes [1]_.
The truncated cube is created by truncating (cutting off) the tips
of the cube one third of the way into each edge [2]_.
parameters:


returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.small.truncated_cube_graph()

**Truncated tetrahedron graph:**
Returns the skeleton of the truncated Platonic tetrahedron.

The truncated tetrahedron is an Archimedean solid with 4 regular hexagonal faces,
4 equilateral triangle faces, 12 nodes and 18 edges. It can be constructed by truncating
all 4 vertices of a regular tetrahedron at one third of the original edge length [1]_.
parameters:


returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.small.truncated_tetrahedron_graph()

**Tutte graph:**
Returns the Tutte graph.

The Tutte graph is a cubic polyhedral, non-Hamiltonian graph. It has
46 nodes and 69 edges.
It is a counterexample to Tait's conjecture that every 3-regular polyhedron
has a Hamiltonian cycle.
It can be realized geometrically from a tetrahedron by multiply truncating
three of its vertices [1]_.
parameters:


returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.small.tutte_graph()
