**Evaluate model:**
Evaluates the given model on the test set using the specified evaluator type.
Args:
    evaluator_type: The type of evaluator to use. Note: When using classification based methods, evaluation may be extremely slow.
    metrics_str: Comma separated list, "ALL" if all metrics are needed.
parameters:
  - model_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': "[].other.*[] | [?type == 'pykeen-model'].key"}] = PyKEENmodel --?
  - evaluator_type: <enum 'EvaluatorTypes'> = EvaluatorTypes.RankBasedEvaluator --?
  - eval_table: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}] = edges_test --?
  - additional_true_triples_table: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}] = edges_train --?
  - metrics_str: <class 'str'> = ALL --?
  - batch_size: <class 'int'> = 32 --?
  - bundle: <class 'lynxkite_graph_analytics.bundle.Bundle'> = ? --?

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.pykeen_ops.evaluate(model_name=<model_name_value>, evaluator_type=<evaluator_type_value>, eval_table=<eval_table_value>, additional_true_triples_table=<additional_true_triples_table_value>, metrics_str=<metrics_str_value>, batch_size=<batch_size_value>, bundle=<bundle_variable>)
