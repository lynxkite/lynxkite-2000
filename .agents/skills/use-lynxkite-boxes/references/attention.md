**Attention:**

parameters:
  - embed_dim: <class 'int'> = 1024 --?
  - num_heads: <class 'int'> = 1 --?
  - dropout: <class 'float'> = ? --?
  - query: <class 'inspect._empty'> = ? --?
  - key: <class 'inspect._empty'> = ? --?
  - value: <class 'inspect._empty'> = ? --?

returns:
  - outputs: ? - ?.
  - weights: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.pytorch.pytorch_ops.attention(embed_dim=<embed_dim_value>, num_heads=<num_heads_value>, dropout=<dropout_value>, query=<query_variable>, key=<key_variable>, value=<value_variable>)
