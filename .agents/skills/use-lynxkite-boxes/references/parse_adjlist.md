**Parse adjlist:**
Parse lines of a graph adjacency list representation.
parameters:
  - comments: str | None = # --Marker for comment lines
  - delimiter: str | None = ? --Separator for node labels.  The default is whitespace.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.readwrite.adjlist.parse_adjlist(comments=<comments_value>, delimiter=<delimiter_value>)
