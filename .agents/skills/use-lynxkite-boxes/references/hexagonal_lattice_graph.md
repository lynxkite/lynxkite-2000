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
