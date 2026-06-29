**Visibility graph:**
Return a Visibility Graph of an input Time Series.

A visibility graph converts a time series into a graph. The constructed graph
uses integer nodes to indicate which event in the series the node represents.
Edges are formed as follows: consider a bar plot of the series and view that
as a side view of a landscape with a node at the top of each bar. An edge
means that the nodes can be connected by a straight "line-of-sight" without
being obscured by any bars between the nodes.

The resulting graph inherits several properties of the series in its structure.
Thereby, periodic series convert into regular graphs, random series convert
into random graphs, and fractal series convert into scale-free networks [1]_.
parameters:


returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.time_series.visibility_graph()
