**Triplet margin loss:**

parameters:
  - x: tensor = ? --?
  - x_pos: tensor = ? --?
  - x_neg: tensor = ? --?

returns:
  - loss: tensor - ?.

usage:
output_variable = lynxkite_core.ops.no_op(x=<x_variable>, x_pos=<x_pos_variable>, x_neg=<x_neg_variable>)
