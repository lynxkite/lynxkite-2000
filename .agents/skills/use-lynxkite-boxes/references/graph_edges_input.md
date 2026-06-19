**Input: graph edges:**
The edges of a graph as input. A 2xE tensor of src/dst indices. Not batched.
parameters:
  - _input_name: <class 'str'> = ? --?

returns:
  - input: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.pytorch.pytorch_ops.graph_edges_input(_input_name=<_input_name_value>)
