---
name: networkx-readwrite-gml
description: Collection of operations - Read GML, Parse GML
---

**Read GML:**
Read graph in GML format from `path`.
parameters:
  - label: str | None = label - .

usage:
output_variable = networkx.readwrite.gml.read_gml(label=<label_value>)

**Parse GML:**
Parse GML graph from a string or iterable.
parameters:
  - label: str | None = label - .

usage:
output_variable = networkx.readwrite.gml.parse_gml(label=<label_value>)
