**Train/test split:**
Splits a dataframe in the bundle into separate "_train" and "_test" dataframes.
parameters:
  - table_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}] = ? --?
  - test_ratio: <class 'float'> = 0.1 --?
  - seed: <class 'int'> = 1234 --?
  - bundle: <class 'lynxkite_graph_analytics.bundle.Bundle'> = ? --?

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.ml_ops.train_test_split(table_name=<table_name_value>, test_ratio=<test_ratio_value>, seed=<seed_value>, bundle=<bundle_variable>)
