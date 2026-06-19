---
name: networkx-readwrite-edgelist
description: Collection of operations - Parse edgelist, Read edgelist, Read weighted edgelist
---

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

**Read edgelist:**
Read a graph from a list of edges.
parameters:
  - comments: str | None = # --The character used to indicate the start of a comment. To specify that
no character should be treated as a comment, use ``comments=None``.
  - delimiter: str | None = ? --The string used to separate values.  The default is whitespace.
  - encoding: str | None = utf-8 --Specify which encoding to use when reading file.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.readwrite.edgelist.read_edgelist(comments=<comments_value>, delimiter=<delimiter_value>, encoding=<encoding_value>)

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
