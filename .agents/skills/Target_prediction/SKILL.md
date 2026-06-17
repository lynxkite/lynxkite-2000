
---
name: Target_prediction
description: Leave the target prediction field empty
---

Leave the target prediction field empty

parameters:
  - bundle: core.Bundle = None
  - model_name: PyKEENModelName = PyKEENmodel
  - head: str = None
  - relation: str = None
  - tail: str = None
  - inductive_setting: bool = False

usage:
output_variable = lynxkite_graph_analytics.src.lynxkite_graph_analytics.operations.pykeen_ops.target_predict(bundle=<bundle_variable>, model_name=<model_name_value>, head=<head_value>, relation=<relation_value>, tail=<tail_value>, inductive_setting=<inductive_setting_value>)

Replace <*_variable> with the output of a previous operation, and <*_value> with a constant value.
