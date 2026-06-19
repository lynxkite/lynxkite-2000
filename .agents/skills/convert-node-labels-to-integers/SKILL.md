---
name: convert-node-labels-to-integers
description: Returns a copy of the graph G with the nodes relabeled using
---

**Convert node labels to integers:**
Returns a copy of the graph G with the nodes relabeled using
consecutive integers.
parameters:
  - first_label: int | None = 0 --An integer specifying the starting offset in numbering nodes.
The new integer labels are numbered first_label, ..., n-1+first_label.
  - ordering: <class 'str'> = default --"default" : inherit node ordering from G.nodes()
"sorted"  : inherit node ordering from sorted(G.nodes())
"increasing degree" : nodes are sorted by increasing degree
"decreasing degree" : nodes are sorted by decreasing degree
  - label_attribute: str | None = ? --Name of node attribute to store old label.  If None no attribute
is created.
  - G: <class 'networkx.classes.graph.Graph'> = ? --A NetworkX graph

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.relabel.convert_node_labels_to_integers(first_label=<first_label_value>, ordering=<ordering_value>, label_attribute=<label_attribute_value>, G=<G_variable>)
