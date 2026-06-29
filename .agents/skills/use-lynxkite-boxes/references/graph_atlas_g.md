**Graph atlas g:**
Returns the list of all graphs with up to seven nodes named in the
Graph Atlas.

The graphs are listed in increasing order by

1. number of nodes,
2. number of edges,
3. degree sequence (for example 111223 < 112222),
4. number of automorphisms,

in that order, with three exceptions as described in the *Notes*
section below. This causes the list to correspond with the index of
the graphs in the Graph Atlas [atlas]_, with the first graph,
``G[0]``, being the null graph.
parameters:


returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.atlas.graph_atlas_g()
