**Bethe–Hessian spectrum:**
Returns eigenvalues of the Bethe Hessian matrix of G.
parameters:
  - r: <class 'float'> = ? --Regularizer parameter
  - G: <class 'networkx.classes.graph.Graph'> = ? --A NetworkX Graph or DiGraph

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.linalg.spectrum.bethe_hessian_spectrum(r=<r_value>, G=<G_variable>)
