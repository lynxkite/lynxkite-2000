**Triples prediction:**

parameters:
  - model_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': "[].other.*[] | [?type == 'pykeen-model'].key"}] = PyKEENmodel --?
  - table_name: typing.Annotated[str, {'format': 'dropdown', 'metadata_query': '[].dataframes[].keys(@)[]'}] = edges_val --?
  - inductive_setting: <class 'bool'> = ? --?
  - bundle: <class 'lynxkite_graph_analytics.bundle.Bundle'> = ? --?

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.pykeen_ops.triple_predict(model_name=<model_name_value>, table_name=<table_name_value>, inductive_setting=<inductive_setting_value>, bundle=<bundle_variable>)
