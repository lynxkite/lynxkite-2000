---
name: train-model
description: Trains the selected model on the selected dataset.
---

Trains the selected model on the selected dataset.
Training parameters specific to the model are set in the model definition,
while parameters specific to the hardware environment and dataset are set here.

parameters:
  - bundle: core.Bundle = None
  - model_name: pytorch_core.PyTorchModelName = model
  - input_mapping: Any = None
  - epochs: int = 1
  - batch_size: int = 1

usage:
output_variable = lynxkite_graph_analytics.operations.ml_ops.train_model(bundle=<bundle_variable>, model_name=<model_name_value>, input_mapping=<input_mapping_value>, epochs=<epochs_value>, batch_size=<batch_size_value>)

Replace <*_variable> with the output of a previous operation, and <*_value> with a constant value.
