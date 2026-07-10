**Neural ODE with MLP:**
A neural ODE for predicting a 1-dimensional value over time, using an MLP to model the derivative.

Must be used with batch size 1.
```python
@op("Neural ODE with MLP", weights=True)
def neural_ode_mlp(
    state_0,
    timestamps,
    *,
    method=ODEMethod.dopri5,
    relative_tolerance=1e-3,
    absolute_tolerance=1e-3,
    state_dimensions=1,
    mlp_layers=3,
    mlp_hidden_size=64,
    mlp_activation=ActivationTypes.ReLU,
):
    """A neural ODE for predicting a 1-dimensional value over time, using an MLP to model the derivative.

    Must be used with batch size 1.
    """
    return ODEWithMLP(
        rtol=relative_tolerance,
        atol=absolute_tolerance,
        input_dim=state_dimensions,
        hidden_dim=mlp_hidden_size,
        num_layers=mlp_layers,
        activation_type=mlp_activation,
        method=method,
    )

```
