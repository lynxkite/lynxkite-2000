---
name: lynxkite-core-ops
description: Collection of operations - Pick element by index, Pick element by constant, Take first n, Drop first n, Graph conv, Heterogeneous graph conv, Triplet margin loss, Cross-entropy loss, Optimizer, Repeat, Recurrent chain
---

**Pick element by index:**

parameters:
  - x: tensor = ? --?
  - index: tensor = ? --?

returns:
  - x_i: tensor - ?.

usage:
output_variable = lynxkite_core.ops.no_op(x=<x_variable>, index=<index_variable>)

**Pick element by constant:**

parameters:
  - index: <class 'str'> = 0 --?
  - x: tensor = ? --?

returns:
  - x_i: tensor - ?.

usage:
output_variable = lynxkite_core.ops.no_op(index=<index_value>, x=<x_variable>)

**Take first n:**

parameters:
  - n: <class 'int'> = 1 --?
  - x: tensor = ? --?

returns:
  - x: tensor - ?.

usage:
output_variable = lynxkite_core.ops.no_op(n=<n_value>, x=<x_variable>)

**Drop first n:**

parameters:
  - n: <class 'int'> = 1 --?
  - x: tensor = ? --?

returns:
  - x: tensor - ?.

usage:
output_variable = lynxkite_core.ops.no_op(n=<n_value>, x=<x_variable>)

**Graph conv:**

parameters:
  - type: <enum 'OptionsFor_type_0f688db2'> = GCNConv --?
  - x: tensor = ? --?
  - edges: tensor = ? --?

returns:
  - x: tensor - ?.

usage:
output_variable = lynxkite_core.ops.no_op(type=<type_value>, x=<x_variable>, edges=<edges_variable>)

**Heterogeneous graph conv:**

parameters:
  - node_embeddings_order: ? = ? --?
  - edge_modules_order: ? = ? --?
  - node_embeddings: tensor = ? --?
  - edge_modules: tensor = ? --?

returns:
  - x: tensor - ?.

usage:
output_variable = lynxkite_core.ops.no_op(node_embeddings_order=<node_embeddings_order_value>, edge_modules_order=<edge_modules_order_value>, node_embeddings=<node_embeddings_variable>, edge_modules=<edge_modules_variable>)

**Triplet margin loss:**

parameters:
  - x: tensor = ? --?
  - x_pos: tensor = ? --?
  - x_neg: tensor = ? --?

returns:
  - loss: tensor - ?.

usage:
output_variable = lynxkite_core.ops.no_op(x=<x_variable>, x_pos=<x_pos_variable>, x_neg=<x_neg_variable>)

**Cross-entropy loss:**

parameters:
  - x: tensor = ? --?
  - y: tensor = ? --?

returns:
  - loss: tensor - ?.

usage:
output_variable = lynxkite_core.ops.no_op(x=<x_variable>, y=<y_variable>)

**Optimizer:**

parameters:
  - type: <enum 'OptionsFor_type_3d5308f5'> = AdamW --?
  - lr: <class 'float'> = 0.0001 --?
  - loss: tensor = ? --?

returns:


usage:
output_variable = lynxkite_core.ops.no_op(type=<type_value>, lr=<lr_value>, loss=<loss_variable>)

**Repeat:**

parameters:
  - times: <class 'int'> = 1 --?
  - same_weights: <class 'bool'> = ? --?
  - input: tensor = ? --?

returns:
  - output: tensor - ?.

usage:
output_variable = lynxkite_core.ops.no_op(times=<times_value>, same_weights=<same_weights_value>, input=<input_variable>)

**Recurrent chain:**

parameters:
  - input: tensor = ? --?

returns:
  - output: tensor - ?.

usage:
output_variable = lynxkite_core.ops.no_op(input=<input_variable>)
