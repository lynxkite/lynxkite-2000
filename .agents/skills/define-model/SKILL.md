---
name: define-model
description: Trains the selected model on the selected dataset. Most training parameters are set in the model definition.
---

Trains the selected model on the selected dataset. Most training parameters are set in the model definition.

parameters:
  - bundle: core.Bundle = None
  - model_workspace: str = None
  - save_as: str = model

usage:
output_variable = lynxkite_graph_analytics.operations.ml_ops.define_model(bundle=<bundle_variable>, model_workspace=<model_workspace_value>, save_as=<save_as_value>)

Replace <*_variable> with the output of a previous operation, and <*_value> with a constant value.
