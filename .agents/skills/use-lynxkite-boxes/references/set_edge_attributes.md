**Set edge attributes:**
Sets edge attributes from a given value or dictionary of values.

.. Warning:: The call order of arguments `values` and `name`
    switched between v1.x & v2.x.
parameters:
  - name: <class 'str'> = ? --Name of the edge attribute to set if values is a scalar.
  - G: <class 'networkx.classes.graph.Graph'> = ? --?
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.classes.function.set_edge_attributes(name=<name_value>, G=<G_variable>)
