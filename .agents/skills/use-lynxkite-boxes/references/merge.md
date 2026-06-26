**Merge:**
Merge multiple inputs
parameters:
  - merge_mode: <enum 'BundleMergeMode'> = must be unique --?
  - bundles: list[lynxkite_graph_analytics.bundle.Bundle] = ? --?

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.graph_ops.merge(merge_mode=<merge_mode_value>, bundles=<bundles_variable>)
