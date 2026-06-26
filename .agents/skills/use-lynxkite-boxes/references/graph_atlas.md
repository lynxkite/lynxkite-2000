**Graph atlas:**
Returns graph number `i` from the Graph Atlas.

For more information, see :func:`.graph_atlas_g`.
parameters:
  - i: <class 'int'> = ? --The index of the graph from the atlas to get. The graph at index
0 is assumed to be the null graph.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.atlas.graph_atlas(i=<i_value>)
