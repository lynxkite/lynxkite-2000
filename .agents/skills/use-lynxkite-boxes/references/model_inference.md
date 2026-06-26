**Model inference:**
Executes a trained model.
parameters:
  - model_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': "[].other.*[] | [?type == 'pytorch-model'].key"}] = model --?
  - input_mapping: lynxkite_graph_analytics.operations.ml_ops.ModelInferenceInputMapping | None = ? --?
  - output_mapping: lynxkite_graph_analytics.operations.ml_ops.ModelOutputMapping | None = ? --?
  - batch_size: <class 'int'> = 1 --?
  - bundle: <class 'lynxkite_graph_analytics.bundle.Bundle'> = ? --?

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.ml_ops.model_inference(model_name=<model_name_value>, input_mapping=<input_mapping_value>, output_mapping=<output_mapping_value>, batch_size=<batch_size_value>, bundle=<bundle_variable>)
