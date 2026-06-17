
---
name: Define_PyKEEN_model
description: Defines a PyKEEN model based on the selected model type.
---

Defines a PyKEEN model based on the selected model type.

parameters:
  - bundle: core.Bundle = None
  - model: PyKEENModelMoreD = PyKEENModelMoreD.MuRE
  - edge_data_table: core.TableName = edges
  - embedding_dim: int = 50
  - loss_function: PyKEENSupportedLosses = PyKEENSupportedLosses.NSSALoss
  - seed: int = 42
  - save_as: str = PyKEENmodel

usage:
output_variable = lynxkite_graph_analytics.src.lynxkite_graph_analytics.operations.pykeen_ops.define_pykeen_model(bundle=<bundle_variable>, model=<model_value>, edge_data_table=<edge_data_table_value>, embedding_dim=<embedding_dim_value>, loss_function=<loss_function_value>, seed=<seed_value>, save_as=<save_as_value>)

Replace <*_variable> with the output of a previous operation, and <*_value> with a constant value.
