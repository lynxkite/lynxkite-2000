
---
name: Merge
description: Merge multiple inputs
---

Merge multiple inputs

parameters:
  - bundles: Any = None
  - merge_mode: bundle.BundleMergeMode = bundle.BundleMergeMode.must_be_unique

usage:
output_variable = lynxkite_graph_analytics.src.lynxkite_graph_analytics.operations.graph_ops.merge(bundles=<bundles_variable>, merge_mode=<merge_mode_value>)

Replace <*_variable> with the output of a previous operation, and <*_value> with a constant value.
