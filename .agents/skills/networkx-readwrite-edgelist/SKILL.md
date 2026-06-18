---
name: networkx-readwrite-edgelist
description: Collection of operations - Parse edgelist, Read edgelist, Read weighted edgelist
---

**Parse edgelist:**
Parse lines of an edge list representation of a graph.
parameters:
  - comments: str | None = # -
  - delimiter: str | None = None -

usage:
output_variable = networkx.readwrite.edgelist.parse_edgelist(comments=<comments_value>, delimiter=<delimiter_value>)

**Read edgelist:**
Read a graph from a list of edges.
parameters:
  - comments: str | None = # -
  - delimiter: str | None = None -
  - encoding: str | None = utf-8 -

usage:
output_variable = networkx.readwrite.edgelist.read_edgelist(comments=<comments_value>, delimiter=<delimiter_value>, encoding=<encoding_value>)

**Read weighted edgelist:**
Read a graph as list of edges with numeric weights.
parameters:
  - comments: str | None = # -
  - delimiter: str | None = None -
  - encoding: str | None = utf-8 -

usage:
output_variable = networkx.readwrite.edgelist.read_weighted_edgelist(comments=<comments_value>, delimiter=<delimiter_value>, encoding=<encoding_value>)
