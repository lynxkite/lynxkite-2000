
---
name: View_vectors
description: View_vectors
---



parameters:
  - bundle: core.Bundle = None
  - table_name: core.TableName = nodes
  - vector_column: core.ColumnNameByTableName =
  - label_column: core.ColumnNameByTableName =
  - n_neighbors: int = 15
  - min_dist: float = 0.1
  - metric: UMAPMetric = UMAPMetric.euclidean

usage:
output_variable = lynxkite_graph_analytics.src.lynxkite_graph_analytics.operations.ml_ops.view_vectors(bundle=<bundle_variable>, table_name=<table_name_value>, vector_column=<vector_column_value>, label_column=<label_column_value>, n_neighbors=<n_neighbors_value>, min_dist=<min_dist_value>, metric=<metric_value>)

Replace <*_variable> with the output of a previous operation, and <*_value> with a constant value.
