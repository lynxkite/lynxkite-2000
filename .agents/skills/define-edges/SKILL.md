---
name: define-edges
description: Define edges between node tables
---

Define edges between node tables

parameters:
  - b: core.Bundle = None
  - relations: str =

usage:
output_variable = lynxkite_graph_analytics.operations.graph_ops.define_edges(b=<b_variable>, relations=<relations_value>)

Replace <*_variable> with the output of a previous operation, and <*_value> with a constant value.
