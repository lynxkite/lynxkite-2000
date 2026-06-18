---
name: convert-node-labels-to-integers
description: Returns a copy of the graph G with the nodes relabeled using
---

**Convert node labels to integers:**
Returns a copy of the graph G with the nodes relabeled using
consecutive integers.
parameters:
  - first_label: int | None = 0 - .
  - ordering: <class 'str'> = default - .
  - label_attribute: str | None = None - .
  - G: <class 'networkx.classes.graph.Graph'> = None - .

usage:
output_variable = networkx.relabel.convert_node_labels_to_integers(first_label=<first_label_value>, ordering=<ordering_value>, label_attribute=<label_attribute_value>, G=<G_variable>)
