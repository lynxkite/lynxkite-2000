**Train embedding model:**

parameters:
  - model: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': "[].other.*[] | [?type == 'pykeen-model'].key"}] = PyKEENmodel --?
  - training_table: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}] = edges_train --?
  - testing_table: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}] = edges_test --?
  - validation_table: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}] = edges_val --?
  - optimizer_type: <enum 'PyKEENSupportedOptimizers'> = PyKEENSupportedOptimizers.Adam --?
  - learning_rate: <class 'float'> = 0.0001 --?
  - epochs: <class 'int'> = 5 --?
  - training_approach: <enum 'TrainingType'> = sLCWA --?
  - number_of_negative_samples_per_positive: <class 'int'> = 512 --?
  - bundle: <class 'lynxkite_graph_analytics.bundle.Bundle'> = ? --?

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.pykeen_ops.train_embedding_model(model=<model_value>, training_table=<training_table_value>, testing_table=<testing_table_value>, validation_table=<validation_table_value>, optimizer_type=<optimizer_type_value>, learning_rate=<learning_rate_value>, epochs=<epochs_value>, training_approach=<training_approach_value>, number_of_negative_samples_per_positive=<number_of_negative_samples_per_positive_value>, bundle=<bundle_variable>)
