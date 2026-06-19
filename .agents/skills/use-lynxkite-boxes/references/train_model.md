**Train model:**
Trains the selected model on the selected dataset.
Training parameters specific to the model are set in the model definition,
while parameters specific to the hardware environment and dataset are set here.
parameters:
  - model_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': "[].other.*[] | [?type == 'pytorch-model'].key"}] = model --?
  - input_mapping: lynxkite_graph_analytics.operations.ml_ops.ModelTrainingInputMapping | None = ? --?
  - epochs: <class 'int'> = 1 --?
  - batch_size: <class 'int'> = 1 --?
  - bundle: <class 'lynxkite_graph_analytics.bundle.Bundle'> = ? --?

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.ml_ops.train_model(model_name=<model_name_value>, input_mapping=<input_mapping_value>, epochs=<epochs_value>, batch_size=<batch_size_value>, bundle=<bundle_variable>)
