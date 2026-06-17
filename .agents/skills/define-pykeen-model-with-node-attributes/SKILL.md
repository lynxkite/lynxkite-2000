---
name: define-pykeen-model-with-node-attributes
description: Defines a PyKEEN model capable of using numeric literals as node attributes.
---

Defines a PyKEEN model capable of using numeric literals as node attributes.

parameters:
  - dataset: core.Bundle = None
  - interaction_name: PyKEENModel1D = PyKEENModel1D.TransE
  - combination_name: PyKEENCombinations = PyKEENCombinations.ConcatProjection
  - embedding_dim: int = None
  - loss_function: str = None
  - random_seed: int = None
  - save_as: str = None

usage:
output_variable = lynxkite_graph_analytics.operations.pykeen_ops.def_pykeen_with_attributes(dataset=<dataset_variable>, interaction_name=<interaction_name_value>, combination_name=<combination_name_value>, embedding_dim=<embedding_dim_value>, loss_function=<loss_function_value>, random_seed=<random_seed_value>, save_as=<save_as_value>)

Replace <*_variable> with the output of a previous operation, and <*_value> with a constant value.
