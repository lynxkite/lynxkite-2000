
---
name: Train-test_split
description: Splits a dataframe in the bundle into separate "_train" and "_test" dataframes.
---

Splits a dataframe in the bundle into separate "_train" and "_test" dataframes.

parameters:
  - bundle: core.Bundle = None
  - table_name: core.TableName = None
  - test_ratio: float = 0.1
  - seed: Any = 1234

usage:
output_variable = lynxkite_graph_analytics.src.lynxkite_graph_analytics.operations.ml_ops.train_test_split(bundle=<bundle_variable>, table_name=<table_name_value>, test_ratio=<test_ratio_value>, seed=<seed_value>)

Replace <*_variable> with the output of a previous operation, and <*_value> with a constant value.
