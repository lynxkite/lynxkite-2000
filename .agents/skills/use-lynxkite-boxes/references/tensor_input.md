**Input: tensor:**
An input tensor.
parameters:
  - _input_name: <class 'str'> = ? --?
  - type: <enum 'TorchTypes'> = float --The data type of the tensor.
  - per_sample: <class 'bool'> = True --Whether this has a different value for each sample, or is constant across the dataset.

returns:
  - input: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.pytorch.pytorch_ops.tensor_input(_input_name=<_input_name_value>, type=<type_value>, per_sample=<per_sample_value>)
