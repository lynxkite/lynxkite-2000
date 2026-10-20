**Activation:**

```python
@op("Activation")
def activation(x, *, type: ActivationTypes = ActivationTypes.ReLU):
    return type.to_layer()

```
