**Bethe–Hessian matrix:**
Returns the Bethe Hessian matrix of G.

The Bethe Hessian is a family of matrices parametrized by r, defined as
H(r) = (r^2 - 1) I - r A + D where A is the adjacency matrix, D is the
diagonal matrix of node degrees, and I is the identify matrix. It is equal
to the graph laplacian when the regularizer r = 1.

The default choice of regularizer should be the ratio [2]_

.. math::
  r_m = \left(\sum k_i \right)^{-1}\left(\sum k_i^2 \right) - 1
parameters:
  - r: <class 'float'> = ? --Regularizer parameter
  - G: <class 'networkx.classes.graph.Graph'> = ? --A NetworkX graph

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.linalg.bethehessianmatrix.bethe_hessian_matrix(r=<r_value>, G=<G_variable>)
