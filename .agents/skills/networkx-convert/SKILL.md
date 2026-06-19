---
name: networkx-convert
description: Collection of operations - From dict of dicts, From dict of lists, From edgelist
---

**From dict of dicts:**
Returns a graph from a dictionary of dictionaries.
parameters:
  - multigraph_input: <class 'bool'> = ? --When True, the dict `d` is assumed
to be a dict-of-dict-of-dict-of-dict structure keyed by
node to neighbor to edge keys to edge data for multi-edges.
Otherwise this routine assumes dict-of-dict-of-dict keyed by
node to neighbor to edge data.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.convert.from_dict_of_dicts(multigraph_input=<multigraph_input_value>)

**From dict of lists:**
Returns a graph from a dictionary of lists.
parameters:


returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.convert.from_dict_of_lists()

**From edgelist:**
Returns a graph from a list of edges.
parameters:


returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.convert.from_edgelist()
