**Chordal cycle graph:**
Returns the chordal cycle graph on `p` nodes.

The returned graph is a cycle graph on `p` nodes with chords joining each
vertex `x` to its inverse modulo `p`. This graph is a (mildly explicit)
3-regular expander [1]_.

`p` *must* be a prime number.
parameters:
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.generators.expanders.chordal_cycle_graph()
