**Define PyKEEN model:**
Defines a PyKEEN model based on the selected model type.
parameters:
  - model: <enum 'PyKEENModelMoreD'> = MuRE --?
  - edge_data_table: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}] = edges --?
  - embedding_dim: <class 'int'> = 50 --?
  - loss_function: <enum 'PyKEENSupportedLosses'> = NSSALoss --?
  - seed: <class 'int'> = 42 --?
  - save_as: <class 'str'> = PyKEENmodel --?
  - bundle: <class 'lynxkite_graph_analytics.bundle.Bundle'> = ? --?

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.pykeen_ops.define_pykeen_model(model=<model_value>, edge_data_table=<edge_data_table_value>, embedding_dim=<embedding_dim_value>, loss_function=<loss_function_value>, seed=<seed_value>, save_as=<save_as_value>, bundle=<bundle_variable>)
