---
name: parse-adjlist
description: Parse lines of a graph adjacency list representation.
---

**Parse adjlist:**
Parse lines of a graph adjacency list representation.
parameters:
  - comments: str | None = # - .
  - delimiter: str | None = None - .

usage:
output_variable = networkx.readwrite.adjlist.parse_adjlist(comments=<comments_value>, delimiter=<delimiter_value>)
