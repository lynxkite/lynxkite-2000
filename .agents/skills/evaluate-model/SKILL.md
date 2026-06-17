---
name: evaluate-model
description: Evaluates the given model on the test set using the specified evaluator type.
---

Evaluates the given model on the test set using the specified evaluator type.
Args:
    evaluator_type: The type of evaluator to use. Note: When using classification based methods, evaluation may be extremely slow.
    metrics_str: Comma separated list, "ALL" if all metrics are needed.

parameters:
  - bundle: core.Bundle = None
  - model_name: PyKEENModelName = PyKEENmodel
  - evaluator_type: EvaluatorTypes = EvaluatorTypes.RankBasedEvaluator
  - eval_table: core.TableName = edges_test
  - additional_true_triples_table: core.TableName = edges_train
  - metrics_str: str = ALL
  - batch_size: int = 32

usage:
output_variable = lynxkite_graph_analytics.operations.pykeen_ops.evaluate(bundle=<bundle_variable>, model_name=<model_name_value>, evaluator_type=<evaluator_type_value>, eval_table=<eval_table_value>, additional_true_triples_table=<additional_true_triples_table_value>, metrics_str=<metrics_str_value>, batch_size=<batch_size_value>)

Replace <*_variable> with the output of a previous operation, and <*_value> with a constant value.
