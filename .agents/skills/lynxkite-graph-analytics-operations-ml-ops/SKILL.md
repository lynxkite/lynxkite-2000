---
name: lynxkite-graph-analytics-operations-ml-ops
description: Collection of operations - Define model, Train model, Model inference, Train/test split, Train/test/validation split, View loss, View vectors
---

**Define model:**
Trains the selected model on the selected dataset. Most training parameters are set in the model definition.
parameters:
  - model_workspace: <class 'str'> = None - .
  - save_as: <class 'str'> = model - .
  - bundle: <class 'lynxkite_graph_analytics.bundle.Bundle'> = None - .

usage:
output_variable = lynxkite_graph_analytics.operations.ml_ops.define_model(model_workspace=<model_workspace_value>, save_as=<save_as_value>, bundle=<bundle_variable>)

**Train model:**
Trains the selected model on the selected dataset.
Training parameters specific to the model are set in the model definition,
while parameters specific to the hardware environment and dataset are set here.
parameters:
  - model_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': "[].other.*[] | [?type == 'pytorch-model'].key"}] = model - .
  - input_mapping: lynxkite_graph_analytics.operations.ml_ops.ModelTrainingInputMapping | None = None - .
  - epochs: <class 'int'> = 1 - .
  - batch_size: <class 'int'> = 1 - .
  - bundle: <class 'lynxkite_graph_analytics.bundle.Bundle'> = None - .

usage:
output_variable = lynxkite_graph_analytics.operations.ml_ops.train_model(model_name=<model_name_value>, input_mapping=<input_mapping_value>, epochs=<epochs_value>, batch_size=<batch_size_value>, bundle=<bundle_variable>)

**Model inference:**
Executes a trained model.
parameters:
  - model_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': "[].other.*[] | [?type == 'pytorch-model'].key"}] = model - .
  - input_mapping: lynxkite_graph_analytics.operations.ml_ops.ModelInferenceInputMapping | None = None - .
  - output_mapping: lynxkite_graph_analytics.operations.ml_ops.ModelOutputMapping | None = None - .
  - batch_size: <class 'int'> = 1 - .
  - bundle: <class 'lynxkite_graph_analytics.bundle.Bundle'> = None - .

usage:
output_variable = lynxkite_graph_analytics.operations.ml_ops.model_inference(model_name=<model_name_value>, input_mapping=<input_mapping_value>, output_mapping=<output_mapping_value>, batch_size=<batch_size_value>, bundle=<bundle_variable>)

**Train/test split:**
Splits a dataframe in the bundle into separate "_train" and "_test" dataframes.
parameters:
  - table_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}] = None - .
  - test_ratio: <class 'float'> = 0.1 - .
  - seed: <class 'int'> = 1234 - .
  - bundle: <class 'lynxkite_graph_analytics.bundle.Bundle'> = None - .

usage:
output_variable = lynxkite_graph_analytics.operations.ml_ops.train_test_split(table_name=<table_name_value>, test_ratio=<test_ratio_value>, seed=<seed_value>, bundle=<bundle_variable>)

**Train/test/validation split:**
Splits a dataframe in the bundle into separate "_train", "_test" and "_val" dataframes.
parameters:
  - table_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}] = None - .
  - test_ratio: <class 'float'> = 0.1 - .
  - val_ratio: <class 'float'> = 0.1 - .
  - seed: <class 'int'> = 1234 - .
  - bundle: <class 'lynxkite_graph_analytics.bundle.Bundle'> = None - .

usage:
output_variable = lynxkite_graph_analytics.operations.ml_ops.train_test_val_split(table_name=<table_name_value>, test_ratio=<test_ratio_value>, val_ratio=<val_ratio_value>, seed=<seed_value>, bundle=<bundle_variable>)

**View loss:**

parameters:
  - bundle: <class 'lynxkite_graph_analytics.bundle.Bundle'> = None - .

usage:
output_variable = lynxkite_graph_analytics.operations.ml_ops.view_loss(bundle=<bundle_variable>)

**View vectors:**

parameters:
  - table_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}] = nodes - .
  - vector_column: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].<table_name>.columns[]'}] =  - .
  - label_column: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].<table_name>.columns[]'}] =  - .
  - n_neighbors: <class 'int'> = 15 - .
  - min_dist: <class 'float'> = 0.1 - .
  - metric: <enum 'UMAPMetric'> = euclidean - .
  - bundle: <class 'lynxkite_graph_analytics.bundle.Bundle'> = None - .

usage:
output_variable = lynxkite_graph_analytics.operations.ml_ops.view_vectors(table_name=<table_name_value>, vector_column=<vector_column_value>, label_column=<label_column_value>, n_neighbors=<n_neighbors_value>, min_dist=<min_dist_value>, metric=<metric_value>, bundle=<bundle_variable>)
