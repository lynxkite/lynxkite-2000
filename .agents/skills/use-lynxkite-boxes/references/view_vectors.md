**View vectors:**

parameters:
  - table_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}] = nodes --?
  - vector_column: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].<table_name>.columns[]'}] = ? --?
  - label_column: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].<table_name>.columns[]'}] = ? --?
  - n_neighbors: <class 'int'> = 15 --?
  - min_dist: <class 'float'> = 0.1 --?
  - metric: <enum 'UMAPMetric'> = euclidean --?
  - bundle: <class 'lynxkite_graph_analytics.bundle.Bundle'> = ? --?

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.ml_ops.view_vectors(table_name=<table_name_value>, vector_column=<vector_column_value>, label_column=<label_column_value>, n_neighbors=<n_neighbors_value>, min_dist=<min_dist_value>, metric=<metric_value>, bundle=<bundle_variable>)
