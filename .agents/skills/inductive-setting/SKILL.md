---
name: inductive-setting
description: inductive-setting
---



parameters:
  - bundle: core.Bundle = None
  - model_name: PyKEENModelName = None
  - inductive_testing_table: core.TableName = None
  - inductive_inference_table: core.TableName = None
  - inductive_validation_table: core.TableName = None
  - metrics_str: str = ALL
  - batch_size: int = 32

usage:
output_variable = lynxkite_graph_analytics.operations.pykeen_ops.eval_inductive_model(bundle=<bundle_variable>, model_name=<model_name_value>, inductive_testing_table=<inductive_testing_table_value>, inductive_inference_table=<inductive_inference_table_value>, inductive_validation_table=<inductive_validation_table_value>, metrics_str=<metrics_str_value>, batch_size=<batch_size_value>)

Replace <*_variable> with the output of a previous operation, and <*_value> with a constant value.
