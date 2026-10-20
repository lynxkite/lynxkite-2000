**Dedensify:**
Compresses neighborhoods around high-degree nodes

Reduces the number of edges to high-degree nodes by adding compressor nodes
that summarize multiple edges of the same type to high-degree nodes (nodes
with a degree greater than a given threshold).  Dedensification also has
the added benefit of reducing the number of edges around high-degree nodes.
The implementation currently supports graphs with a single edge type.
parameters:
  - threshold: <class 'int'> = ? --Minimum degree threshold of a node to be considered a high degree node.
The threshold must be greater than or equal to 2.
  - copy: bool | None = ? --Indicates if dedensification should be done inplace
  - G: <class 'networkx.classes.graph.Graph'> = ? --A networkx graph
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.algorithms.summarization.dedensify(threshold=<threshold_value>, copy=<copy_value>, G=<G_variable>)
