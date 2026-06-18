---
name: networkx-classes-function
description: Collection of operations - Set node attributes, Set edge attributes, Is weighted, Is negatively weighted, Is empty
---

**Set node attributes:**
Sets node attributes from a given value or dictionary of values.

.. Warning:: The call order of arguments `values` and `name`
    switched between v1.x & v2.x.
parameters:
  - name: <class 'str'> = None - .
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.classes.function.set_node_attributes(name=<name_value>, G=<G_variable>)

**Set edge attributes:**
Sets edge attributes from a given value or dictionary of values.

.. Warning:: The call order of arguments `values` and `name`
    switched between v1.x & v2.x.
parameters:
  - name: <class 'str'> = None - .
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.classes.function.set_edge_attributes(name=<name_value>, G=<G_variable>)

**Is weighted:**
Returns True if `G` has weighted edges.
parameters:
  - weight: str | None = weight - .
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.classes.function.is_weighted(weight=<weight_value>, G=<G_variable>)

**Is negatively weighted:**
Returns True if `G` has negatively weighted edges.
parameters:
  - weight: str | None = weight - .
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.classes.function.is_negatively_weighted(weight=<weight_value>, G=<G_variable>)

**Is empty:**
Returns True if `G` has no edges.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.classes.function.is_empty(G=<G_variable>)
