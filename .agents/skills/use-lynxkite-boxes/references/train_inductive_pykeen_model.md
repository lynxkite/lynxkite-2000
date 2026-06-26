**Train inductive model:**

parameters:
  - model_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': "[].other.*[] | [?type == 'pykeen-model'].key"}] = ? --?
  - transductive_table_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}] = ? --?
  - inductive_inference_table: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}] = ? --?
  - inductive_validation_table: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}] = ? --?
  - optimizer_type: <enum 'PyKEENSupportedOptimizers'> = PyKEENSupportedOptimizers.Adam --?
  - epochs: <class 'int'> = 5 --?
  - training_approach: <enum 'TrainingType'> = sLCWA --?
  - bundle: <class 'lynxkite_graph_analytics.bundle.Bundle'> = ? --?

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.pykeen_ops.train_inductive_pykeen_model(model_name=<model_name_value>, transductive_table_name=<transductive_table_name_value>, inductive_inference_table=<inductive_inference_table_value>, inductive_validation_table=<inductive_validation_table_value>, optimizer_type=<optimizer_type_value>, epochs=<epochs_value>, training_approach=<training_approach_value>, bundle=<bundle_variable>)
