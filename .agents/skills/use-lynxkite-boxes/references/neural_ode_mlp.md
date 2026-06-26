**Neural ODE with MLP:**
A neural ODE for predicting a 1-dimensional value over time, using an MLP to model the derivative.

Must be used with batch size 1.
parameters:
  - method: <enum 'ODEMethod'> = dopri5 --?
  - relative_tolerance: <class 'float'> = 0.001 --?
  - absolute_tolerance: <class 'float'> = 0.001 --?
  - state_dimensions: <class 'int'> = 1 --?
  - mlp_layers: <class 'int'> = 3 --?
  - mlp_hidden_size: <class 'int'> = 64 --?
  - mlp_activation: <enum 'ActivationTypes'> = ReLU --?
  - state_0: <class 'inspect._empty'> = ? --?
  - timestamps: <class 'inspect._empty'> = ? --?

returns:
  - output: ? - ?.

usage:
output_variable = lynxkite_graph_analytics.pytorch.pytorch_ops.neural_ode_mlp(method=<method_value>, relative_tolerance=<relative_tolerance_value>, absolute_tolerance=<absolute_tolerance_value>, state_dimensions=<state_dimensions_value>, mlp_layers=<mlp_layers_value>, mlp_hidden_size=<mlp_hidden_size_value>, mlp_activation=<mlp_activation_value>, state_0=<state_0_variable>, timestamps=<timestamps_variable>)
