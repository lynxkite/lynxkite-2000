**Set node attributes:**
Sets node attributes from a given value or dictionary of values.

.. Warning:: The call order of arguments `values` and `name`
    switched between v1.x & v2.x.
parameters:
  - name: <class 'str'> = ? --Name of the node attribute to set if values is a scalar.
  - G: <class 'networkx.classes.graph.Graph'> = ? --?
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.classes.function.set_node_attributes(name=<name_value>, G=<G_variable>)
