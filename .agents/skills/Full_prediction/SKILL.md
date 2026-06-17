
---
name: Full_prediction
description: Warning: This prediction can be a very expensive operation!
---

Warning: This prediction can be a very expensive operation!

Args:
    k: Pass "" to keep all scores

parameters:
  - bundle: core.Bundle = None
  - model_name: PyKEENModelName = PyKEENmodel
  - k: Any = None
  - inductive_setting: bool = False

usage:
output_variable = lynxkite_graph_analytics.src.lynxkite_graph_analytics.operations.pykeen_ops.full_predict(bundle=<bundle_variable>, model_name=<model_name_value>, k=<k_value>, inductive_setting=<inductive_setting_value>)

Replace <*_variable> with the output of a previous operation, and <*_value> with a constant value.
