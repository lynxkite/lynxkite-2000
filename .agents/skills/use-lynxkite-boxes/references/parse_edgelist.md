**Parse edgelist:**
Parse lines of an edge list representation of a graph.
parameters:
  - comments: str | None = # --Marker for comment lines. Default is `'#'`. To specify that no character
should be treated as a comment, use ``comments=None``.
  - delimiter: str | None = ? --Separator for node labels. Default is `None`, meaning any whitespace.
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.readwrite.edgelist.parse_edgelist(comments=<comments_value>, delimiter=<delimiter_value>)
