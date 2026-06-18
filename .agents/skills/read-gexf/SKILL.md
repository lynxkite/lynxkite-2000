---
name: read-gexf
description: Read graph in GEXF format from path.
---

**Read gexf:**
Read graph in GEXF format from path.

"GEXF (Graph Exchange XML Format) is a language for describing
complex networks structures, their associated data and dynamics" [1]_.
parameters:
  - relabel: <class 'bool'> = None -
  - version: <class 'str'> = 1.2draft -

usage:
output_variable = networkx.readwrite.gexf.read_gexf(relabel=<relabel_value>, version=<version_value>)
