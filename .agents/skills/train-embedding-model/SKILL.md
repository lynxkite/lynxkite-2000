---
name: train-embedding-model
description: train-embedding-model
---



parameters:
  - bundle: core.Bundle = None
  - model: PyKEENModelName = PyKEENmodel
  - training_table: core.TableName = edges_train
  - testing_table: core.TableName = edges_test
  - validation_table: core.TableName = edges_val
  - optimizer_type: PyKEENSupportedOptimizers = PyKEENSupportedOptimizers.Adam
  - learning_rate: float = 0.0001
  - epochs: int = 5
  - training_approach: TrainingType = TrainingType.sLCWA
  - number_of_negative_samples_per_positive: int = 512

usage:
output_variable = lynxkite_graph_analytics.operations.pykeen_ops.train_embedding_model(bundle=<bundle_variable>, model=<model_value>, training_table=<training_table_value>, testing_table=<testing_table_value>, validation_table=<validation_table_value>, optimizer_type=<optimizer_type_value>, learning_rate=<learning_rate_value>, epochs=<epochs_value>, training_approach=<training_approach_value>, number_of_negative_samples_per_positive=<number_of_negative_samples_per_positive_value>)

Replace <*_variable> with the output of a previous operation, and <*_value> with a constant value.
