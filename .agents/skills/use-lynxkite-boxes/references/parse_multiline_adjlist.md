**Parse multiline adjlist:**
Parse lines of a multiline adjacency list representation of a graph.
parameters:
  - comments: str | None = # --Marker for comment lines
  - delimiter: str | None = ? --Separator for node labels.  The default is whitespace.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.readwrite.multiline_adjlist.parse_multiline_adjlist(comments=<comments_value>, delimiter=<delimiter_value>)
