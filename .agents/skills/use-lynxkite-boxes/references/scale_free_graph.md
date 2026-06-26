**Scale-free graph:**
Returns a scale-free directed graph.
parameters:
  - n: <class 'int'> = ? --Number of nodes in graph
  - alpha: <class 'float'> = 0.41 --Probability for adding a new node connected to an existing node
chosen randomly according to the in-degree distribution.
  - beta: <class 'float'> = 0.54 --Probability for adding an edge between two existing nodes.
One existing node is chosen randomly according the in-degree
distribution and the other chosen randomly according to the out-degree
distribution.
  - gamma: <class 'float'> = 0.05 --Probability for adding a new node connected to an existing node
chosen randomly according to the out-degree distribution.
  - delta_in: <class 'float'> = 0.2 --Bias for choosing nodes from in-degree distribution.
  - delta_out: <class 'float'> = 0 --Bias for choosing nodes from out-degree distribution.
  - seed: int | None = ? --Indicator of random number generation state.
See :ref:`Randomness<randomness>`.

returns:
  - output: <class 'networkx.classes.graph.Graph'> - ?.

usage:
output_variable = networkx.generators.directed.scale_free_graph(n=<n_value>, alpha=<alpha_value>, beta=<beta_value>, gamma=<gamma_value>, delta_in=<delta_in_value>, delta_out=<delta_out_value>, seed=<seed_value>)
