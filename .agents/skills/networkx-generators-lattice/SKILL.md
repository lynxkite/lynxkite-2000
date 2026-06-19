---
name: networkx-generators-lattice
description: Collection of operations - Grid 2D graph, Grid graph, Hypercube graph, Triangular lattice graph, Hexagonal lattice graph
---

**Grid 2D graph:**
Returns the two-dimensional grid graph.

The grid graph has each node connected to its four nearest neighbors.
parameters:


returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.lattice.grid_2d_graph()

**Grid graph:**
Returns the *n*-dimensional grid graph.

The dimension *n* is the length of the list `dim` and the size in
each dimension is the value of the corresponding list element.
parameters:


returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.lattice.grid_graph()

**Hypercube graph:**
Returns the *n*-dimensional hypercube graph.

The *n*-dimensional hypercube graph [1]_ has ``2**n`` nodes, each represented as
a binary integer in the form of a tuple of 0's and 1's. Edges exist between
nodes that differ in exactly one bit.
parameters:
  - n: <class 'int'> = ? --Dimension of the hypercube, must be a positive integer.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.lattice.hypercube_graph(n=<n_value>)

**Triangular lattice graph:**
Returns the $m$ by $n$ triangular lattice graph.

The `triangular lattice graph`_ is a two-dimensional `grid graph`_ in
which each square unit has a diagonal edge (each grid unit has a chord).

The returned graph has $m$ rows and $n$ columns of triangles. Rows and
columns include both triangles pointing up and down. Rows form a strip
of constant height. Columns form a series of diamond shapes, staggered
with the columns on either side. Another way to state the size is that
the nodes form a grid of `m+1` rows and `(n + 1) // 2` columns.
The odd row nodes are shifted horizontally relative to the even rows.

Directed graph types have edges pointed up or right.

Positions of nodes are computed by default or `with_positions is True`.
The position of each node (embedded in a euclidean plane) is stored in
the graph using equilateral triangles with sidelength 1.
The height between rows of nodes is thus $\sqrt(3)/2$.
Nodes lie in the first quadrant with the node $(0, 0)$ at the origin.

.. _triangular lattice graph: http://mathworld.wolfram.com/TriangularGrid.html
.. _grid graph: http://www-cs-students.stanford.edu/~amitp/game-programming/grids/
.. _Triangular Tiling: https://en.wikipedia.org/wiki/Triangular_tiling
parameters:
  - m: <class 'int'> = ? --The number of rows in the lattice.
  - n: <class 'int'> = ? --The number of columns in the lattice.
  - periodic: <class 'bool'> = ? --If True, join the boundary vertices of the grid using periodic
boundary conditions. The join between boundaries is the final row
and column of triangles. This means there is one row and one column
fewer nodes for the periodic lattice. Periodic lattices require
`m >= 3`, `n >= 5` and are allowed but misaligned if `m` or `n` are odd
  - with_positions: <class 'bool'> = ? --Store the coordinates of each node in the graph node attribute 'pos'.
The coordinates provide a lattice with equilateral triangles.
Periodic positions shift the nodes vertically in a nonlinear way so
the edges don't overlap so much.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.lattice.triangular_lattice_graph(m=<m_value>, n=<n_value>, periodic=<periodic_value>, with_positions=<with_positions_value>)

**Hexagonal lattice graph:**
Returns an `m` by `n` hexagonal lattice graph.

The *hexagonal lattice graph* is a graph whose nodes and edges are
the `hexagonal tiling`_ of the plane.

The returned graph will have `m` rows and `n` columns of hexagons.
`Odd numbered columns`_ are shifted up relative to even numbered columns.

Positions of nodes are computed by default or `with_positions is True`.
Node positions creating the standard embedding in the plane
with sidelength 1 and are stored in the node attribute 'pos'.
`pos = nx.get_node_attributes(G, 'pos')` creates a dict ready for drawing.

.. _hexagonal tiling: https://en.wikipedia.org/wiki/Hexagonal_tiling
.. _Odd numbered columns: http://www-cs-students.stanford.edu/~amitp/game-programming/grids/
parameters:
  - m: <class 'int'> = ? --The number of rows of hexagons in the lattice.
  - n: <class 'int'> = ? --The number of columns of hexagons in the lattice.
  - periodic: <class 'bool'> = ? --Whether to make a periodic grid by joining the boundary vertices.
For this to work `n` must be even and both `n > 1` and `m > 1`.
The periodic connections create another row and column of hexagons
so these graphs have fewer nodes as boundary nodes are identified.
  - with_positions: <class 'bool'> = ? --Store the coordinates of each node in the graph node attribute 'pos'.
The coordinates provide a lattice with vertical columns of hexagons
offset to interleave and cover the plane.
Periodic positions shift the nodes vertically in a nonlinear way so
the edges don't overlap so much.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.lattice.hexagonal_lattice_graph(m=<m_value>, n=<n_value>, periodic=<periodic_value>, with_positions=<with_positions_value>)
