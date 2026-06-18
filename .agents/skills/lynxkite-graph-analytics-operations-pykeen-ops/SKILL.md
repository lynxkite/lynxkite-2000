---
name: lynxkite-graph-analytics-operations-pykeen-ops
description: Collection of operations - Import PyKEEN dataset, Import inductive dataset, Split inductive dataset, Define PyKEEN model, Define PyKEEN model with node attributes, Define inductive PyKEEN model, Train embedding model, Train inductive model, View early stopping metric, Triples prediction, Target prediction, Full prediction, Extract embeddings from PyKEEN model, Evaluate model, Evaluate inductive model
---

**Import PyKEEN dataset:**
Imports a dataset from the PyKEEN library.
parameters:
  - dataset: <enum 'PyKEENDataset'> = PyKEENDataset.Nations - .

usage:
output_variable = lynxkite_graph_analytics.operations.pykeen_ops.import_pykeen_dataset_path(dataset=<dataset_value>)

**Import inductive dataset:**
Imports an inductive dataset from the PyKEEN library.
parameters:
  - dataset: <enum 'InductiveDataset'> = InductiveDataset.ILPC2022Small - .

usage:
output_variable = lynxkite_graph_analytics.operations.pykeen_ops.import_inductive_dataset(dataset=<dataset_value>)

**Split inductive dataset:**
Splits incoming data into 4 subsets. Transductive training on which training should be run, inductive inference on which during training inference is done.
Inference testing and validation sets that can be used to evaluate model performance.
parameters:
  - dataset_table: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}] = None - .
  - entity_ratio: <class 'float'> = 0.5 - .
  - training_ratio: <class 'float'> = 0.8 - .
  - testing_ratio: <class 'float'> = 0.1 - .
  - validation_ratio: <class 'float'> = 0.1 - .
  - seed: <class 'int'> = 42 - .
  - bundle: <class 'lynxkite_graph_analytics.bundle.Bundle'> = None - .

usage:
output_variable = lynxkite_graph_analytics.operations.pykeen_ops.inductively_split_dataset(dataset_table=<dataset_table_value>, entity_ratio=<entity_ratio_value>, training_ratio=<training_ratio_value>, testing_ratio=<testing_ratio_value>, validation_ratio=<validation_ratio_value>, seed=<seed_value>, bundle=<bundle_variable>)

**Define PyKEEN model:**
Defines a PyKEEN model based on the selected model type.
parameters:
  - model: <enum 'PyKEENModelMoreD'> = MuRE - .
  - edge_data_table: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}] = edges - .
  - embedding_dim: <class 'int'> = 50 - .
  - loss_function: <enum 'PyKEENSupportedLosses'> = NSSALoss - .
  - seed: <class 'int'> = 42 - .
  - save_as: <class 'str'> = PyKEENmodel - .
  - bundle: <class 'lynxkite_graph_analytics.bundle.Bundle'> = None - .

usage:
output_variable = lynxkite_graph_analytics.operations.pykeen_ops.define_pykeen_model(model=<model_value>, edge_data_table=<edge_data_table_value>, embedding_dim=<embedding_dim_value>, loss_function=<loss_function_value>, seed=<seed_value>, save_as=<save_as_value>, bundle=<bundle_variable>)

**Define PyKEEN model with node attributes:**
Defines a PyKEEN model capable of using numeric literals as node attributes.
parameters:
  - interaction_name: <enum 'PyKEENModel1D'> = TransE - .
  - combination_name: <enum 'PyKEENCombinations'> = PyKEENCombinations.ConcatProjection - .
  - embedding_dim: <class 'int'> = None - .
  - loss_function: <class 'str'> = None - .
  - random_seed: <class 'int'> = None - .
  - save_as: <class 'str'> = None - .
  - combination_group: group = PyKEENCombinations.ConcatProjection - .
  - dataset: <class 'lynxkite_graph_analytics.bundle.Bundle'> = None - .

usage:
output_variable = lynxkite_graph_analytics.operations.pykeen_ops.def_pykeen_with_attributes(interaction_name=<interaction_name_value>, combination_name=<combination_name_value>, embedding_dim=<embedding_dim_value>, loss_function=<loss_function_value>, random_seed=<random_seed_value>, save_as=<save_as_value>, combination_group=<combination_group_value>, dataset=<dataset_variable>)

**Define inductive PyKEEN model:**
Defines an InductiveNodePiece model (with an optional GNN message passing layer) for inductive link prediction tasks.
parameters:
  - triples_table: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}] = None - .
  - inference_table: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}] = None - .
  - interaction: <enum 'PyKEENModel1D'> = DistMult - .
  - embedding_dim: <class 'int'> = 200 - .
  - loss_function: <class 'str'> = None - .
  - num_tokens: <class 'int'> = 2 - .
  - aggregation: <enum 'PyTorchAggregationFunctions'> = PyTorchAggregationFunctions.MLP - .
  - use_GNN: <class 'bool'> = False - .
  - seed: <class 'int'> = 42 - .
  - save_as: <class 'str'> = InductiveModel - .
  - bundle: <class 'lynxkite_graph_analytics.bundle.Bundle'> = None - .

usage:
output_variable = lynxkite_graph_analytics.operations.pykeen_ops.get_inductive_model(triples_table=<triples_table_value>, inference_table=<inference_table_value>, interaction=<interaction_value>, embedding_dim=<embedding_dim_value>, loss_function=<loss_function_value>, num_tokens=<num_tokens_value>, aggregation=<aggregation_value>, use_GNN=<use_GNN_value>, seed=<seed_value>, save_as=<save_as_value>, bundle=<bundle_variable>)

**Train embedding model:**

parameters:
  - model: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': "[].other.*[] | [?type == 'pykeen-model'].key"}] = PyKEENmodel - .
  - training_table: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}] = edges_train - .
  - testing_table: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}] = edges_test - .
  - validation_table: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}] = edges_val - .
  - optimizer_type: <enum 'PyKEENSupportedOptimizers'> = PyKEENSupportedOptimizers.Adam - .
  - learning_rate: <class 'float'> = 0.0001 - .
  - epochs: <class 'int'> = 5 - .
  - training_approach: <enum 'TrainingType'> = sLCWA - .
  - number_of_negative_samples_per_positive: <class 'int'> = 512 - .
  - bundle: <class 'lynxkite_graph_analytics.bundle.Bundle'> = None - .

usage:
output_variable = lynxkite_graph_analytics.operations.pykeen_ops.train_embedding_model(model=<model_value>, training_table=<training_table_value>, testing_table=<testing_table_value>, validation_table=<validation_table_value>, optimizer_type=<optimizer_type_value>, learning_rate=<learning_rate_value>, epochs=<epochs_value>, training_approach=<training_approach_value>, number_of_negative_samples_per_positive=<number_of_negative_samples_per_positive_value>, bundle=<bundle_variable>)

**Train inductive model:**

parameters:
  - model_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': "[].other.*[] | [?type == 'pykeen-model'].key"}] = None - .
  - transductive_table_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}] = None - .
  - inductive_inference_table: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}] = None - .
  - inductive_validation_table: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}] = None - .
  - optimizer_type: <enum 'PyKEENSupportedOptimizers'> = PyKEENSupportedOptimizers.Adam - .
  - epochs: <class 'int'> = 5 - .
  - training_approach: <enum 'TrainingType'> = sLCWA - .
  - bundle: <class 'lynxkite_graph_analytics.bundle.Bundle'> = None - .

usage:
output_variable = lynxkite_graph_analytics.operations.pykeen_ops.train_inductive_pykeen_model(model_name=<model_name_value>, transductive_table_name=<transductive_table_name_value>, inductive_inference_table=<inductive_inference_table_value>, inductive_validation_table=<inductive_validation_table_value>, optimizer_type=<optimizer_type_value>, epochs=<epochs_value>, training_approach=<training_approach_value>, bundle=<bundle_variable>)

**View early stopping metric:**

parameters:
  - bundle: <class 'lynxkite_graph_analytics.bundle.Bundle'> = None - .

usage:
output_variable = lynxkite_graph_analytics.operations.pykeen_ops.view_early_stopping(bundle=<bundle_variable>)

**Triples prediction:**

parameters:
  - model_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': "[].other.*[] | [?type == 'pykeen-model'].key"}] = PyKEENmodel - .
  - table_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}] = edges_val - .
  - inductive_setting: <class 'bool'> = False - .
  - bundle: <class 'lynxkite_graph_analytics.bundle.Bundle'> = None - .

usage:
output_variable = lynxkite_graph_analytics.operations.pykeen_ops.triple_predict(model_name=<model_name_value>, table_name=<table_name_value>, inductive_setting=<inductive_setting_value>, bundle=<bundle_variable>)

**Target prediction:**
Leave the target prediction field empty
parameters:
  - model_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': "[].other.*[] | [?type == 'pykeen-model'].key"}] = PyKEENmodel - .
  - head: <class 'str'> = None - .
  - relation: <class 'str'> = None - .
  - tail: <class 'str'> = None - .
  - inductive_setting: <class 'bool'> = False - .
  - bundle: <class 'lynxkite_graph_analytics.bundle.Bundle'> = None - .

usage:
output_variable = lynxkite_graph_analytics.operations.pykeen_ops.target_predict(model_name=<model_name_value>, head=<head_value>, relation=<relation_value>, tail=<tail_value>, inductive_setting=<inductive_setting_value>, bundle=<bundle_variable>)

**Full prediction:**
Warning: This prediction can be a very expensive operation!
parameters:
  - model_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': "[].other.*[] | [?type == 'pykeen-model'].key"}] = PyKEENmodel - .
  - k: int | None = None - .
  - inductive_setting: <class 'bool'> = False - .
  - bundle: <class 'lynxkite_graph_analytics.bundle.Bundle'> = None - .

usage:
output_variable = lynxkite_graph_analytics.operations.pykeen_ops.full_predict(model_name=<model_name_value>, k=<k_value>, inductive_setting=<inductive_setting_value>, bundle=<bundle_variable>)

**Extract embeddings from PyKEEN model:**

parameters:
  - model_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': "[].other.*[] | [?type == 'pykeen-model'].key"}] = PyKEENmodel - .
  - bundle: <class 'lynxkite_graph_analytics.bundle.Bundle'> = None - .

usage:
output_variable = lynxkite_graph_analytics.operations.pykeen_ops.extract_from_pykeen(model_name=<model_name_value>, bundle=<bundle_variable>)

**Evaluate model:**
Evaluates the given model on the test set using the specified evaluator type.
Args:
    evaluator_type: The type of evaluator to use. Note: When using classification based methods, evaluation may be extremely slow.
    metrics_str: Comma separated list, "ALL" if all metrics are needed.
parameters:
  - model_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': "[].other.*[] | [?type == 'pykeen-model'].key"}] = PyKEENmodel - .
  - evaluator_type: <enum 'EvaluatorTypes'> = EvaluatorTypes.RankBasedEvaluator - .
  - eval_table: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}] = edges_test - .
  - additional_true_triples_table: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}] = edges_train - .
  - metrics_str: <class 'str'> = ALL - .
  - batch_size: <class 'int'> = 32 - .
  - bundle: <class 'lynxkite_graph_analytics.bundle.Bundle'> = None - .

usage:
output_variable = lynxkite_graph_analytics.operations.pykeen_ops.evaluate(model_name=<model_name_value>, evaluator_type=<evaluator_type_value>, eval_table=<eval_table_value>, additional_true_triples_table=<additional_true_triples_table_value>, metrics_str=<metrics_str_value>, batch_size=<batch_size_value>, bundle=<bundle_variable>)

**Evaluate inductive model:**

parameters:
  - model_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': "[].other.*[] | [?type == 'pykeen-model'].key"}] = None - .
  - inductive_testing_table: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}] = None - .
  - inductive_inference_table: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}] = None - .
  - inductive_validation_table: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}] = None - .
  - metrics_str: <class 'str'> = ALL - .
  - batch_size: <class 'int'> = 32 - .
  - bundle: <class 'lynxkite_graph_analytics.bundle.Bundle'> = None - .

usage:
output_variable = lynxkite_graph_analytics.operations.pykeen_ops.eval_inductive_model(model_name=<model_name_value>, inductive_testing_table=<inductive_testing_table_value>, inductive_inference_table=<inductive_inference_table_value>, inductive_validation_table=<inductive_validation_table_value>, metrics_str=<metrics_str_value>, batch_size=<batch_size_value>, bundle=<bundle_variable>)
