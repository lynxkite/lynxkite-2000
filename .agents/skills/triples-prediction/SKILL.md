---
name: triples-prediction
description: triples-prediction
---



parameters:
  - bundle: core.Bundle = None
  - model_name: PyKEENModelName = PyKEENmodel
  - table_name: core.TableName = edges_val
  - inductive_setting: bool = False

usage:
output_variable = lynxkite_graph_analytics.operations.pykeen_ops.triple_predict(bundle=<bundle_variable>, model_name=<model_name_value>, table_name=<table_name_value>, inductive_setting=<inductive_setting_value>)

Replace <*_variable> with the output of a previous operation, and <*_value> with a constant value.
