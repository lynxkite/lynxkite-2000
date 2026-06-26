**Full prediction:**
Warning: This prediction can be a very expensive operation!
parameters:
  - model_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': "[].other.*[] | [?type == 'pykeen-model'].key"}] = PyKEENmodel --?
  - k: int | None = ? --Pass "" to keep all scores
  - inductive_setting: <class 'bool'> = ? --?
  - bundle: <class 'lynxkite_graph_analytics.bundle.Bundle'> = ? --?

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.pykeen_ops.full_predict(model_name=<model_name_value>, k=<k_value>, inductive_setting=<inductive_setting_value>, bundle=<bundle_variable>)
