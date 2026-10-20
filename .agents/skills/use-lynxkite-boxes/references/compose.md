**Compose:**
Compose graph G with H by combining nodes and edges into a single graph.

The node sets and edges sets do not need to be disjoint.

Composing preserves the attributes of nodes and edges.
Attribute values from H take precedent over attribute values from G.
parameters:
  - G: <class 'networkx.classes.graph.Graph'> = ? --A NetworkX graph
  - H: <class 'networkx.classes.graph.Graph'> = ? --A NetworkX graph
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.operators.binary.compose(G=<G_variable>, H=<H_variable>)
