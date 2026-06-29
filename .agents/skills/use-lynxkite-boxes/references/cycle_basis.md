**Cycle basis:**
Returns a list of cycles which form a basis for cycles of G.

A basis for cycles of a network is a minimal collection of
cycles such that any cycle in the network can be written
as a sum of cycles in the basis.  Here summation of cycles
is defined as "exclusive or" of the edges. Cycle bases are
useful, e.g. when deriving equations for electric circuits
using Kirchhoff's Laws.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --?

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.algorithms.cycles.cycle_basis(G=<G_variable>)
