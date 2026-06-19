**Target prediction:**
Leave the target prediction field empty
parameters:
  - model_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': "[].other.*[] | [?type == 'pykeen-model'].key"}] = PyKEENmodel --?
  - head: <class 'str'> = ? --?
  - relation: <class 'str'> = ? --?
  - tail: <class 'str'> = ? --?
  - inductive_setting: <class 'bool'> = ? --?
  - bundle: <class 'lynxkite_graph_analytics.bundle.Bundle'> = ? --?

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.pykeen_ops.target_predict(model_name=<model_name_value>, head=<head_value>, relation=<relation_value>, tail=<tail_value>, inductive_setting=<inductive_setting_value>, bundle=<bundle_variable>)
