**Evaluate inductive model:**

parameters:
  - model_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': "[].other.*[] | [?type == 'pykeen-model'].key"}] = ? --?
  - inductive_testing_table: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}] = ? --?
  - inductive_inference_table: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}] = ? --?
  - inductive_validation_table: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}] = ? --?
  - metrics_str: <class 'str'> = ALL --?
  - batch_size: <class 'int'> = 32 --?
  - bundle: <class 'lynxkite_graph_analytics.bundle.Bundle'> = ? --?

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.pykeen_ops.eval_inductive_model(model_name=<model_name_value>, inductive_testing_table=<inductive_testing_table_value>, inductive_inference_table=<inductive_inference_table_value>, inductive_validation_table=<inductive_validation_table_value>, metrics_str=<metrics_str_value>, batch_size=<batch_size_value>, bundle=<bundle_variable>)
