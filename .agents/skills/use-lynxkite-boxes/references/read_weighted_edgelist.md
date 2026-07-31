**Read weighted edgelist:**
Read a graph as list of edges with numeric weights.
parameters:
  - comments: str | None = # --The character used to indicate the start of a comment.
  - delimiter: str | None = ? --The string used to separate values.  The default is whitespace.
  - encoding: str | None = utf-8 --Specify which encoding to use when reading file.
returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.
usage:
  output_variable = networkx.readwrite.edgelist.read_weighted_edgelist(comments=<comments_value>, delimiter=<delimiter_value>, encoding=<encoding_value>)
