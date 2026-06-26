**Extract embeddings from PyKEEN model:**

parameters:
  - model_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': "[].other.*[] | [?type == 'pykeen-model'].key"}] = PyKEENmodel --?
  - bundle: <class 'lynxkite_graph_analytics.bundle.Bundle'> = ? --?

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.pykeen_ops.extract_from_pykeen(model_name=<model_name_value>, bundle=<bundle_variable>)
