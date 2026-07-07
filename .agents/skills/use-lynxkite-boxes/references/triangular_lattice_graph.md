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
