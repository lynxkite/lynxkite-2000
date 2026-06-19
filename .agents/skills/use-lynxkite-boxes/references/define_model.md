**Define model:**
Trains the selected model on the selected dataset. Most training parameters are set in the model definition.
parameters:
  - model_workspace: <class 'str'> = ? --?
  - save_as: <class 'str'> = model --?
  - bundle: <class 'lynxkite_graph_analytics.bundle.Bundle'> = ? --?

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.ml_ops.define_model(model_workspace=<model_workspace_value>, save_as=<save_as_value>, bundle=<bundle_variable>)
