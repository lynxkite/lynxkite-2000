**Define PyKEEN model with node attributes:**
Defines a PyKEEN model capable of using numeric literals as node attributes.
parameters:
  - interaction_name: <enum 'PyKEENModel1D'> = TransE --?
  - combination_name: <enum 'PyKEENCombinations'> = PyKEENCombinations.ConcatProjection --?
  - embedding_dim: <class 'int'> = ? --?
  - loss_function: <class 'str'> = ? --?
  - random_seed: <class 'int'> = ? --?
  - save_as: <class 'str'> = ? --?
  - combination_group: group = PyKEENCombinations.ConcatProjection --?
  - dataset: <class 'lynxkite_graph_analytics.bundle.Bundle'> = ? --?

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.operations.pykeen_ops.def_pykeen_with_attributes(interaction_name=<interaction_name_value>, combination_name=<combination_name_value>, embedding_dim=<embedding_dim_value>, loss_function=<loss_function_value>, random_seed=<random_seed_value>, save_as=<save_as_value>, combination_group=<combination_group_value>, dataset=<dataset_variable>)
