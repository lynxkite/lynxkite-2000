---
name: model-inference
description: Executes a trained model.
---

Executes a trained model.

parameters:
  - bundle: core.Bundle = None
  - model_name: pytorch_core.PyTorchModelName = model
  - input_mapping: Any = None
  - output_mapping: Any = None
  - batch_size: int = 1

usage:
output_variable = lynxkite_graph_analytics.operations.ml_ops.model_inference(bundle=<bundle_variable>, model_name=<model_name_value>, input_mapping=<input_mapping_value>, output_mapping=<output_mapping_value>, batch_size=<batch_size_value>)

Replace <*_variable> with the output of a previous operation, and <*_value> with a constant value.
